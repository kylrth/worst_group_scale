import argparse
import datetime
import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

from ilc import get_grads as ilc_grads
from ilc import get_train_loader
import utils


def setup_dataset(dataset: str, groups, root_dir: str):
    dataset = get_dataset(dataset=dataset, root_dir=root_dir)

    train = dataset.get_subset(
        "train",
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
    )
    val = dataset.get_subset(
        "val", transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    )

    grouper = CombinatorialGrouper(dataset, groups)

    return train, val, grouper


def init_model(model_name="resnet18", pretrained=False, model_cache="", device="cuda:0"):
    if model_cache != "":
        torch.hub.set_dir(model_cache)

    model = getattr(models, model_name)(pretrained=pretrained)

    # replace the last layer with a random-init layer of the output size we want (1)
    # We can't use num_classes=1 in the model factory above, because pretrained models don't come in
    # that size so it won't know how to init the last layer.
    if model_name == "resnet50":
        model.fc = nn.Linear(2048, 1)
    elif model_name.startswith("resnet"):
        model.fc = nn.Linear(512, 1)
    elif model_name.startswith("vit_b"):
        model.heads[-1] = nn.Linear(768, 1)
    elif model_name.startswith("vit_l"):
        model.heads[-1] = nn.Linear(1024, 1)

    model.train()

    return model.to(device)


def train(config: utils.TrainConfig, model, trainset, valset, recorder, device="cuda:0"):
    # set up optimization
    bce_loss = nn.BCELoss(reduction="none")
    optim = torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )
    sched = StepLR(optim, step_size=config.lr_step)

    # data loading
    trainloader = get_train_loader(
        "standard", trainset, batch_size=config.batch_size, drop_last=config.objective == "ilc"
    )
    valloader = get_eval_loader(
        "standard", valset, batch_size=config.batch_size, drop_last=config.objective == "ilc"
    )

    # look for checkpoints
    ep = recorder.latest_checkpoint(model, optim, sched, device)
    if ep == 0:
        print("starting with fresh model")
        # run evaluation once at the beginning
        validation(model, valloader, 0, bce_loss, recorder, device)
    else:
        print("starting from checkpointed epoch", ep)

    # train
    model.train()
    t = tqdm(initial=ep * len(trainloader), total=config.epochs * len(trainloader), mininterval=10)
    while ep < config.epochs:
        print("starting epoch", ep + 1)
        for i, (x, y, metadata) in enumerate(trainloader):
            it = ep * len(trainloader) + i
            train_iteration(
                config, model, optim, x, y, metadata, ep, it, bce_loss, recorder, device
            )
            t.update()

        sched.step()

        ep += 1

        validation(model, valloader, ep, bce_loss, recorder, device)
        recorder.checkpoint(ep, model, optim, sched)

    recorder.close()
    t.close()


def train_iteration(config, model, optim, x, y, metadata, epoch, it, loss, recorder, device):
    z = recorder.grouper.metadata_to_group(metadata)
    optim.zero_grad()

    # This is left over from supporting grad accumulation, which I removed because it won't work
    # with ILC as we've got it currently, and the GPUs are big enough for the other experiments
    # anyway.
    actual_batch = len(x) // config.grad_accum_factor

    # collect group losses/counts across entire batch
    agg_loss = 0.0
    agg_penalty = 0.0
    losses = [0.0] * recorder.n_groups
    counts = [0] * recorder.n_groups

    for idx in range(0, len(x), actual_batch):
        ex = x[idx : idx + actual_batch].to(device)
        why = y[idx : idx + actual_batch].float().to(device)
        zee = z[idx : idx + actual_batch].to(device)
        logits = torch.sigmoid(model(ex))

        # accumulate group losses/counts (this is just straight-up loss, not necessarily the
        # training objective)
        batch_loss = loss(logits, why.unsqueeze(-1))
        agg_loss += torch.sum(batch_loss).item()
        for i in range(recorder.n_groups):
            mask = zee.eq(i).unsqueeze(-1)
            losses[i] += torch.sum(batch_loss * mask).item()
            counts[i] += torch.sum(mask).item()

        # This is the training objective.
        p = penalty(config, loss, logits, why, batch_loss, epoch, optim)
        if config.objective != "ilc":
            # We don't call backward if it's ILC because the penalty function actually sets
            # the gradients itself.
            p.backward()
        agg_penalty += p.item()

    recorder.report_train(it, agg_loss, losses, counts, agg_penalty)

    optim.step()


def penalty(
    config: utils.TrainConfig, loss: nn.Module, logits, y, batch_loss, epoch: int, optim
) -> torch.Tensor:
    """Returns the training loss for the objective specified by the config."""
    if config.objective == "erm":
        return torch.sum(batch_loss) / config.batch_size

    if config.objective == "irm":
        scale = logits.new_tensor(1.0, requires_grad=True)  # place on same device as logits
        l = loss(logits * scale, y.unsqueeze(-1))
        grad = torch.autograd.grad(l.mean(), [scale], create_graph=True)[0]
        p = torch.sum(grad**2)

        l = torch.sum(l) / config.batch_size

        # see https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py#L145
        p_weight = config.irm_weight if epoch >= config.irm_anneal else 1.0
        l += p_weight * p
        if p_weight > 1.0:
            # keep gradients in a reasonable range
            l /= p_weight
        return l

    if config.objective == "ilc":
        l, _ = ilc_grads(
            agreement_threshold=config.ilc_agreement_threshold,
            batch_size=1,
            loss_fn=loss,
            n_agreement_envs=len(y),
            params=optim.param_groups[0]["params"],
            output=logits,
            target=y,
            method="and_mask",
            scale_grad_inverse_sparsity=1.0,
        )
        return l

    raise NotImplementedError(f"unrecognized penalty type {config.objective}")


def validation(model, valloader, epoch, loss, recorder, device="cuda:0"):
    print("running validation for epoch", epoch)
    with torch.no_grad():
        model.eval()

        losses = [0.0] * recorder.n_groups
        accs = [0.0] * recorder.n_groups

        for x, y, metadata in valloader:
            x = x.to(device)
            y = y.to(device)
            z = recorder.grouper.metadata_to_group(metadata).to(device)

            logits = torch.sigmoid(model(x))
            batch_loss = loss(logits, y.unsqueeze(-1).float())
            batch_preds = logits >= 0.5
            acc = batch_preds.squeeze(-1) == y

            for i in range(recorder.n_groups):
                mask = z.eq(i)
                losses[i] += torch.sum(batch_loss.squeeze(-1) * mask).detach().item()
                accs[i] += torch.logical_and(acc, mask).sum().detach().item()

        recorder.report_valid(epoch, losses, accs)

        model.train()


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def main(exp, wilds_dir, model_cache, tensorboard_dir, checkpoint_dir):
    # This part of the file is meant to be edited to add the sorts of experiments you'd like to run.
    # I'm not a fan of computing the experiment parameters from the SLURM array ID in bash. So we'll
    # define a list of experiments here and let the SLURM array ID index them.
    exps = list(
        itertools.product(
            ["celebA", "waterbirds"],
            ["erm"],
            ["resnet18", "resnet34", "resnet50"],
            list(range(1, 11)),  # seed
        )
    )

    if exp == -1:
        for i, e in enumerate(exps):
            print(f"{i}: {e}")
        return

    # select the experiment according to the SLURM array ID
    exps = exps[exp : exp + 1]
    print(f"running experiment {exp}: {exps[0]}")

    device = get_device()
    print("training with", device)

    for ds, obj, m, seed in exps:
        config = utils.TrainConfig(
            dataset=ds,
            model_name=m,
            pretrained=True,
            epochs=30 if ds == "celebA" else 100,
            objective=obj,
            lr=0.0005,
            seed=seed,
        )

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        trainset, valset, grouper = setup_dataset(
            config.dataset, config.dataset_groups(), wilds_dir
        )
        model = init_model(config.model_name, config.pretrained, model_cache, device)
        recorder = utils.Recorder(config, valset, grouper, tensorboard_dir, checkpoint_dir)
        train(config, model, trainset, valset, recorder, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp",
        metavar="EXP",
        type=int,
        default=-1,
        nargs="?",
        help="experiment number to run. If -1, instead prints the experiment combinations.",
    )
    parser.add_argument(
        "--wilds", type=str, default="./data/", help="path where WILDS data can be stored"
    )
    parser.add_argument(
        "--model-cache", type=str, default="", help="override the PyTorch model cache"
    )
    parser.add_argument(
        "--tensorboard", type=str, default="./runs/", help="path where Tensorboard output is stored"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="path where checkpointed models are stored",
    )

    args = parser.parse_args()

    try:
        main(
            args.exp,
            *(
                os.path.expanduser(p)
                for p in [args.wilds, args.model_cache, args.tensorboard, args.checkpoints]
            ),
        )
    except RuntimeError as err:
        # provide more information for discovering GPU issues
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        instance = utils.get_instance_id()
        print(
            f"ERROR at {ts} on {instance} {get_device()}: {type(err).__name__}: {err}", flush=True
        )
        raise err
