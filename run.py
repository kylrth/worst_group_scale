import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds.common.grouper import CombinatorialGrouper

import utils


def setup_dataset(dataset: str, groups, root_dir: str):
    dataset = get_dataset(dataset=dataset, root_dir=root_dir, download=True)

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

    model = getattr(models, model_name)(pretrained=pretrained, num_classes=1)

    return model.to(device)


def train(config: utils.TrainConfig, model, trainset, valset, recorder, device="cuda:0"):
    # set up optimization
    bce_loss = nn.BCELoss(reduction="none")
    optim = torch.optim.SGD(
        model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )
    sched = StepLR(optim, step_size=config.lr_step)

    # data loading
    trainloader = get_train_loader("standard", trainset, batch_size=config.batch_size)
    valloader = get_eval_loader("standard", valset, batch_size=config.batch_size)

    # look for checkpoints
    ep = recorder.latest_checkpoint(model, optim, sched)
    if ep == 0:
        print("starting with fresh model")
        # run evaluation once at the beginning
        validation(model, valloader, 0, bce_loss, recorder, device)
    else:
        print("starting from checkpointed epoch", ep)

    # train
    t = tqdm(initial=ep, total=config.epochs * len(trainloader))
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

    t.close()
    recorder.close()


def train_iteration(config, model, optim, x, y, metadata, epoch, it, loss, recorder, device):
    z = recorder.grouper.metadata_to_group(metadata)
    optim.zero_grad()

    while True:
        try:
            actual_batch = len(x) // config.grad_accum_factor

            # collect group losses/counts across entire batch
            agg_loss = 0.0
            agg_penalty = 0.0
            losses = [0.0] * recorder.n_groups
            penalties = [0.0] * recorder.n_groups
            counts = [0] * recorder.n_groups

            for idx in range(0, len(x), actual_batch):
                ex = x[idx : idx + actual_batch].to(device)
                why = y[idx : idx + actual_batch].to(device)
                zee = z[idx : idx + actual_batch].to(device)

                logits = torch.sigmoid(model(ex))
                batch_loss, batch_penalty = penalty(
                    config.objective, loss, logits, why.unsqueeze(-1).float()
                )

                # accumulate group losses/counts
                for i in range(recorder.n_groups):
                    mask = zee.eq(i).unsqueeze(-1)
                    losses[i] += torch.sum(batch_loss * mask).item()
                    penalties[i] += torch.sum(batch_penalty * mask).item()
                    counts[i] += torch.sum(mask).item()

                l = torch.sum(batch_loss) / config.batch_size
                p = torch.sum(batch_penalty) / config.batch_size

                # see https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py#L145
                p_weight = config.penalty_weight if epoch >= config.penalty_anneal else 1.0
                l += p_weight * p
                if p_weight > 1.0:
                    # keep gradients in a reasonable range
                    l /= p_weight

                l.backward()
                agg_loss += l.item()
                agg_penalty += p.item()

            recorder.report_train(it, agg_loss, agg_penalty, losses, penalties, counts)

            break
        except RuntimeError as e:
            if not str(e).startswith("CUDA out of memory."):
                raise

            torch.cuda.empty_cache()
            config.grad_accum_factor = config.grad_accum_factor * 2
            print("OOM! increasing gradient accumulation factor to", config.grad_accum_factor)
    optim.step()


def penalty(penalty_type: str, loss: nn.Module, logits, y):
    """Returns the loss and the penalty."""
    if penalty_type.lower() == "erm":
        l = loss(logits, y)
        return l, torch.zeros_like(l)
    if penalty_type.lower() == "irm":
        scale = logits.new(1.0).requires_grad_()
        l = loss(logits * scale, y)
        grad = torch.autograd.grad(l, [scale], create_graph=True)[0]
        return l, torch.sum(grad**2)

    raise NotImplementedError(f"unrecognized penalty type {penalty_type}")


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


def main(exp, wilds_dir, model_cache, tensorboard_dir, checkpoint_dir):
    # This part of the file is meant to be edited to add the sorts of experiments you'd like to run.
    # I'm not a fan of computing the experiment parameters from the SLURM array ID in bash. So we'll
    # define a list of experiments here and let the SLURM array ID index them.
    exps = []
    for model_name in ["resnet18", "vit_b_16"]:
        exps.append(model_name)

    if exp != "all":
        # select the experiment according to the SLURM array ID
        exps = exps[exp : exp + 1]
        print(f"running experiment {exp}: {exps[0]}")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print("training with", device)

    for model_name in exps:
        config = utils.TrainConfig(
            dataset="celebA",
            model_name=model_name,
        )

        trainset, valset, grouper = setup_dataset(
            config.dataset, config.dataset_groups(), wilds_dir
        )
        model = init_model(
            config.model_name, config.pretrained, device=device, model_cache=model_cache
        )
        recorder = utils.Recorder(
            config, valset, grouper, tb_base=tensorboard_dir, ckpt_base=checkpoint_dir
        )
        train(config, model, trainset, valset, recorder, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp", metavar="EXP", type=int, default="all", nargs="?", help="experiment number to run"
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

    main(
        args.exp,
        *(
            os.path.expanduser(p)
            for p in [args.wilds, args.model_cache, args.tensorboard, args.checkpoints]
        ),
    )
