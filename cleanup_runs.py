"""This is a script to clean up any extra lines in the CSV output files that were created by
restarting experiments. This means extra header lines, as well as repeat iterations. (Earlier
entries are superceded by newer ones.)"""

import glob
import sys


def clean(path):
    with open(path) as f:
        lines = f.readlines()

    if "avg" not in lines[0]:
        print("file does not have header on first line: " + path)
        print("skipping...")
        return

    # start with the header and then go through the rest of the lines
    out = [lines[0]]
    last_epoch = -1
    for i, line in enumerate(lines[1:]):
        if "avg" in line:
            # header
            continue

        # get the epoch from the line
        try:
            epoch = int(line[: line.index(",")])
        except ValueError:
            print(f"file has a strange line (line {i+1}): " + path)
            print("skipping...")
            return

        # If the log skips an epoch we'll have to take a look manually.
        if epoch > last_epoch + 1:
            print(f"file skips from epoch {last_epoch} to epoch {epoch}: " + path)
            print("skipping...")
            return

        # find out if we've backtracked and get rid of the past lines
        if epoch < last_epoch + 1:
            out = out[: epoch + 1]  # keep only the header + the first `epoch` lines
        last_epoch = epoch

        out.append(line)

    with open(path, "w") as f:
        f.writelines(out)


def main(to_clean):
    for file in to_clean:
        clean(file)


if __name__ == "__main__":
    main(sys.argv[1:])
