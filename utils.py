from dataclasses import dataclass
import glob
import os

import torch
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainConfig:
    dataset: str = "celebA"
    model_name: str = "resnet18"
    pretrained: bool = False
    batch_size: int = 128
    epochs: int = 30
    objective: str = "erm"
    irm_weight: float = 100000
    irm_anneal: int = 10
    ilc_agreement_threshold: float = 0.3
    lr: float = 0.01
    momentum: float = 0.9
    lr_step: int = 30
    weight_decay: float = 1e-4
    seed: int = 0
    grad_accum_factor: int = 1

    def exp_id(self):
        return repr(self)[12:-1]  # remove "TrainConfig(...)" wrap

    def dataset_groups(self):
        if self.dataset == "celebA":
            return ["male", "y"]
        elif self.dataset == "waterbirds":
            return ["background", "y"]

        raise NotImplementedError(f"unknown dataset {self.dataset}")


class Recorder:
    board: SummaryWriter
    ckpt_dir: str
    discrete_hparams = {
        "dataset": ["celebA", "waterbirds"],
        "model_name": [
            "resnet18",
            "resnet34",
            "resnet50",
            "vit_b_16",
            "vit_b_32",
            "vit_l_16",
            "vit_l_32",
        ],
        "objective": ["erm", "irm", "ilc"],
    }

    def __init__(
        self,
        config: TrainConfig,
        valset,
        grouper,
        tb_base: str = "./runs",
        ckpt_base: str = "./checkpoints",
    ):
        tb_base = os.path.expanduser(tb_base)
        ckpt_base = os.path.expanduser(ckpt_base)

        exp_id = config.exp_id()

        # count the number of groups and the valset group counts
        self.n_groups = grouper.cardinality.prod().item()
        self.valcounts = count_groups(valset, grouper)
        self.grouper = grouper

        self.hparams = {
            "dataset": config.dataset,
            "model_name": config.model_name,
            "pretrained": config.pretrained,
            "batch_size": config.batch_size,
            "objective": config.objective,
            "lr": config.lr,
            "momentum": config.momentum,
            "lr_step": config.lr_step,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
        }
        if config.objective == "irm":
            self.hparams.update(
                {
                    "irm_weight": config.irm_weight,
                    "irm_anneal": config.irm_anneal,
                }
            )
        elif config.objective == "ilc":
            self.hparams["ilc_agreement_threshold"] = config.ilc_agreement_threshold

        os.makedirs(tb_base, exist_ok=True)
        self.board = SummaryWriter(os.path.join(tb_base, exp_id))

        self.ckpt_dir = os.path.join(ckpt_base, exp_id)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def report_train(self, it, agg_loss, agg_penalty, losses, penalties, counts):
        """Report train losses across the groups."""
        self.board.add_scalars(
            "train loss",
            {
                self.grouper.group_str(idx): losses[idx] / counts[idx]
                for idx in range(self.n_groups)
                if counts[idx] > 0
            },
            it,
        )
        self.board.add_scalars("train loss", {"avg": agg_loss}, it)

        self.board.add_scalars(
            "train penalty",
            {
                self.grouper.group_str(idx): penalties[idx] / counts[idx]
                for idx in range(self.n_groups)
                if counts[idx] > 0
            },
            it,
        )
        self.board.add_scalars("train penalty", {"avg": agg_penalty}, it)

    def close(self):
        self.board.close()

    def report_valid(self, epoch, losses, accs):
        """Report validation losses and accuracies across the groups."""
        val_losses = {
            self.grouper.group_str(idx): losses[idx] / self.valcounts[idx]
            for idx in range(self.n_groups)
            if self.valcounts[idx] > 0
        }
        val_losses["avg"] = sum(losses) / sum(self.valcounts)
        self.board.add_scalars("val loss", val_losses, epoch)

        val_accs = {
            self.grouper.group_str(idx): accs[idx] / self.valcounts[idx]
            for idx in range(self.n_groups)
            if self.valcounts[idx] > 0
        }
        val_accs["avg"] = sum(accs) / sum(self.valcounts)
        self.board.add_scalars("val acc", val_accs, epoch)

        self.hparams["epochs"] = epoch
        metrics = {
            "hparam/val_loss": val_losses["avg"],
            "hparam/val_acc": val_accs["avg"],
        }
        for k in val_losses:
            metrics["hparam/val_loss_" + k] = val_losses[k]
        for k in val_accs:
            metrics["hparam/val_acc_" + k] = val_accs[k]
        self.board.add_hparams(self.hparams, metrics, self.discrete_hparams)

        self.board.flush()

    def checkpoint(self, epoch, model, optim, sched):
        """Store a checkpoint for a completed epoch."""
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
            },
            os.path.join(self.ckpt_dir, f"{epoch}.pt"),
        )

    def latest_checkpoint(self, model, optim, sched) -> int:
        """Find the latest checkpoint and load the state. Return the last completed epoch."""
        # find the latest checkpoint by number
        latest_ckpt = 0
        for f in glob.glob(os.path.join(self.ckpt_dir, "*.pt")):
            num = int(f[:-3])  # remove .pt
            if num > latest_ckpt:
                latest_ckpt = num

        if latest_ckpt == 0:
            # no checkpoints
            return 0

        # load the checkpoint
        ckpt = torch.load(os.path.join(self.ckpt_dir, f"{latest_ckpt}.pt"))
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        sched.load_state_dict(ckpt["sched"])

        return latest_ckpt  # completed epoch


def count_groups(subset, grouper):
    z = grouper.metadata_to_group(subset.metadata_array)

    # assumes all groups are represented!
    return z.unique(sorted=True, return_counts=True)[1]
