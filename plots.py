import glob
import os

from matplotlib import pyplot as plt
import pandas as pd

from utils import TrainConfig


def get_experiments(paths):
    for path in paths:
        yield TrainConfig.from_exp_id(os.path.basename(path))


def dataset_title(dataset: str) -> str:
    if dataset.lower() == "celeba":
        return "CelebA"

    return dataset.title()


def triple_plot(csvs, labels, worst_group_label, title, filename):
    fig, ax = plt.subplots(1, 3, figsize=(25, 8))
    for csv, label in zip(csvs, labels):
        ax[0].plot(csv.index, csv["avg"].ewm(alpha=0.1).mean(), label=label)
        ax[1].plot(csv.index, csv[worst_group_label].ewm(alpha=0.1).mean(), label=label)
        ax[2].plot(
            csv.index,
            (csv[worst_group_label] / csv["avg"]).ewm(alpha=0.1).mean(),
            label=label,
        )
    ax[0].legend(loc="lower right")
    ax[2].legend()  # loc="lower right")
    for a in ax:
        a.set_ylim((0, 1))
        a.set_xlabel("epoch")
    ax[0].set_title("average accuracy")
    ax[1].set_title("worst group accuracy")
    ax[2].set_title("disparity ratio")
    fig.suptitle(title)
    plt.savefig(filename, bbox_inches="tight", transparent=True, pad_inches=0)


def main():
    os.makedirs("plots", exist_ok=True)

    # hardcoded values
    model_type = "ResNet"
    sizes = [18, 34, 50]
    obj = "ERM"
    pre = True
    lr = 0.0005

    for dataset in ["celebA", "waterbirds"]:
        if dataset == "waterbirds":
            worst_group_label = "y = waterbird, background =  land"
            epochs = 100
        else:
            worst_group_label = "y =     blond, male = 1"
            epochs = 30

        csvs = []
        for size in sizes:
            # get the mean across seeds
            exps = list(
                get_experiments(
                    glob.glob(
                        f"runs/{dataset},{model_type.lower()}{size},pretrained={pre},batch_size=128,epochs={epochs},objective={obj.lower()},lr={lr},momentum=0.9,lr_step=30,weight_decay=0.0001,seed=*"
                    )
                )
            )

            individs = []
            for exp in exps:
                individ = pd.read_csv(
                    os.path.join("runs", exp.exp_id(), "val_acc.csv"), index_col="epoch"
                )
                individs.append(individ)
            csv = pd.concat(individs)

            csvs.append(csv.groupby(csv.index).mean())

        triple_plot(
            csvs=csvs,
            labels=[model_type + str(size) for size in sizes],
            worst_group_label=worst_group_label,
            title=f"Training {model_type}s with {obj} on {dataset_title(dataset)}"
            + ("" if not pre else " (pretrained)"),
            filename=f"plots/{dataset}_{model_type.lower()}_{obj.lower()}"
            + ("" if not pre else "_pretrained")
            + ".png",
        )


if __name__ == "__main__":
    main()
