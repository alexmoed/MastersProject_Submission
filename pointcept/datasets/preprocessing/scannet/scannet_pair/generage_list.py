#!/usr/bin/env python3
# @brief [Brief description of what the code does]
# Xiaoyang Wu (2023). Pointcept: A Codebase for Point Cloud Perception Research [online].
# [Accessed 2025]. Available from: "https://github.com/Pointcept/Pointcept".
# Original Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)

import argparse
import glob, os, sys

from SensorData import SensorData

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument("--target_dir", required=True, help="path to the target dir")

opt = parser.parse_args()
print(opt)


def main():
    overlaps = glob.glob(os.path.join(opt.target_dir, "*/pcd/overlap.txt"))
    with open(os.path.join(opt.target_dir, "overlap30.txt"), "w") as f:
        for fo in overlaps:
            for line in open(fo):
                pcd0, pcd1, op = line.strip().split()
                if float(op) >= 0.3:
                    print("{} {} {}".format(pcd0, pcd1, op), file=f)
    print("done")


if __name__ == "__main__":
    main()
