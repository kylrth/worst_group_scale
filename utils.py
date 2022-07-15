from dataclasses import dataclass
import glob
import os

import requests
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
        out = (
            f"{self.dataset},{self.model_name},pretrained={self.pretrained},"
            f"batch_size={self.batch_size},epochs={self.epochs},objective={self.objective}"
        )

        if self.objective == "irm":
            out += f",irm_weight={self.irm_weight},irm_anneal={self.irm_anneal}"
        elif self.objective == "ilc":
            out += f",ilc_agreement_threshold={self.ilc_agreement_threshold}"

        out += (
            f",lr={self.lr},momentum={self.momentum},lr_step={self.lr_step}"
            f",weight_decay={self.weight_decay},seed={self.seed}"
        )

        return out

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
        self.exp_id = config.exp_id()
        self.tb_dir = os.path.join(os.path.expanduser(tb_base), self.exp_id)
        self.ckpt_dir = os.path.join(os.path.expanduser(ckpt_base), self.exp_id)

        # count the number of groups and the valset group counts
        self.n_groups = grouper.cardinality.prod().item()
        self.valcounts = count_groups(valset, grouper)
        self.grouper = grouper

        self.config = config
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
        self.board = SummaryWriter(self.tb_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # this is the header we'll use for all the output CSV files
        self.train_csv_header = ["iter", "avg", "penalty"]
        self.valid_csv_header = ["epoch", "avg"]
        for idx in range(self.n_groups):
            self.train_csv_header.append(self.grouper.group_str(idx))
            self.valid_csv_header.append(self.grouper.group_str(idx))

        # we'll append to the files in case we're starting from a checkpoint
        self.train_loss_f = open(os.path.join(self.tb_dir, "train_loss.csv"), "a+")
        self.val_loss_f = open(os.path.join(self.tb_dir, "val_loss.csv"), "a+")
        self.val_acc_f = open(os.path.join(self.tb_dir, "val_acc.csv"), "a+")

        # write the headers
        train_header = ",".join(f'"{k}"' if "," in k else k for k in self.train_csv_header) + "\n"
        valid_header = ",".join(f'"{k}"' if "," in k else k for k in self.valid_csv_header) + "\n"
        self.train_loss_f.write(train_header)
        self.val_loss_f.write(valid_header)
        self.val_acc_f.write(valid_header)

    def _dict_to_csv(self, d, is_valid: bool) -> str:
        if is_valid:
            header = self.valid_csv_header
        else:
            header = self.train_csv_header
        return (
            ",".join(
                str(d[k].item()) if isinstance(d[k], torch.Tensor) else str(d[k]) for k in header
            )
            + "\n"
        )

    def report_train(self, it, agg_loss, group_losses, counts, agg_penalty):
        """Report train losses across the groups."""
        loss_dict = {}
        for idx in range(self.n_groups):
            if counts[idx] <= 0:
                loss_dict[self.grouper.group_str(idx)] = -1.0
                continue
            loss_dict[self.grouper.group_str(idx)] = group_losses[idx] / counts[idx]
        loss_dict["avg"] = agg_loss / self.config.batch_size
        # Unlike the loss, the penalty is already aggregated so we take the average over gradient
        # accumulation runs.
        loss_dict["penalty"] = agg_penalty / self.config.grad_accum_factor

        self.board.add_scalars("train loss", loss_dict, it)
        loss_dict["iter"] = it
        self.train_loss_f.write(self._dict_to_csv(loss_dict, False))

    def report_valid(self, epoch, losses, accs):
        """Report validation losses and accuracies across the groups."""
        val_losses = {}
        for idx in range(self.n_groups):
            if self.valcounts[idx] <= 0:
                val_losses[self.grouper.group_str(idx)] = -1.0
                continue
            val_losses[self.grouper.group_str(idx)] = losses[idx] / self.valcounts[idx]
        val_losses["avg"] = sum(losses) / sum(self.valcounts)

        val_accs = {}
        for idx in range(self.n_groups):
            if self.valcounts[idx] <= 0:
                val_accs[self.grouper.group_str(idx)] = -1.0
                continue
            val_accs[self.grouper.group_str(idx)] = accs[idx] / self.valcounts[idx]
        val_accs["avg"] = sum(accs) / sum(self.valcounts)

        self.board.add_scalars("val loss", val_losses, epoch)
        self.board.add_scalars("val acc", val_accs, epoch)
        val_losses["epoch"] = epoch
        val_accs["epoch"] = epoch
        self.val_loss_f.write(self._dict_to_csv(val_losses, True))
        self.val_acc_f.write(self._dict_to_csv(val_accs, True))

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

    def close(self):
        self.board.close()
        self.train_loss_f.close()
        self.val_loss_f.close()
        self.val_acc_f.close()

    def checkpoint(self, epoch, model, optim, sched):
        """Store a checkpoint for a completed epoch. Only keep the last five checkpoints."""
        if epoch > 5:
            try:
                os.remove(os.path.join(self.ckpt_dir, f"{epoch-5}.pt"))
            except OSError:
                # it's ok if it doesn't exist
                pass

        torch.save(
            {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
            },
            os.path.join(self.ckpt_dir, f"{epoch}.pt"),
        )

    def latest_checkpoint(self, model, optim, sched, device) -> int:
        """Find the latest checkpoint and load the state. Return the last completed epoch."""
        # find the latest checkpoint by number
        latest_ckpt = 0
        for f in glob.glob(os.path.join(self.ckpt_dir, "*.pt")):
            num = int(os.path.basename(f)[:-3])  # remove .pt
            if num > latest_ckpt:
                latest_ckpt = num

        if latest_ckpt == 0:
            # no checkpoints
            return 0

        # load the checkpoint
        ckpt = torch.load(os.path.join(self.ckpt_dir, f"{latest_ckpt}.pt"), map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        sched.load_state_dict(ckpt["sched"])

        return latest_ckpt  # completed epoch


def count_groups(subset, grouper):
    z = grouper.metadata_to_group(subset.metadata_array)

    # assumes all groups are represented!
    return z.unique(sorted=True, return_counts=True)[1]


def get_instance_id() -> str:
    """On StabilityAI infrastructure this will return the instance ID. Otherwise returns "???"."""
    try:
        resp = requests.get("http://169.254.169.254/latest/meta-data/instance-id")
    except Exception:
        # likely not on StabilityAI infrastructure
        return "???"

    return resp.text
