########################################################################################################################
# This script constructs a .csv file containing matching bf/mask paths in each row.                                   #
# Author:               Daniel Schirmacher                                                                             #
#                       PhD Student, Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                   #
# Date:                 01.02.2022                                                                                     #
# Python:               3.8.6                                                                                          #
########################################################################################################################
import argparse
import glob
import os

import pandas as pd


def arg_parse():
    """
    Catch user input.


    Parameter
    ---------

    -


    Return
    ------

    Returns a namespace from `argparse.parse_args()`.
    """
    desc = "Program to obtain a .csv file containing matching bf/mask paths in each row."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--bf",
        type=str,
        default="T:/TimelapseData/180619AW12/*/*w00.png",
        help="Path (glob pattern) to bright field images. Naming convention must match naming "
        "naming convention of mask s.t. sort results in matching paths in each row.",
    )

    parser.add_argument(
        "--mask",
        type=str,
        default="T:/TimelapseData/180619AW12/Analysis/Segmentation_201106/*/*.png",
        help="Path (glob pattern) to segmentation masks.",
    )

    parser.add_argument(
        "--prefix", type=str, default="201201SK30", help="Prefix for output file name."
    )

    parser.add_argument(
        "--out",
        type=str,
        default="C:/Users/schidani/Desktop/",
        help="Path to output directory.",
    )

    return parser.parse_args()


def main():
    args = arg_parse()

    path_bf = args.bf
    path_mask = args.mask
    prefix = args.prefix
    out_path = args.out

    files_bf = glob.glob(path_bf)
    files_bf.sort()
    files_mask = glob.glob(path_mask)
    files_mask.sort()

    df = pd.DataFrame({"bf": files_bf, "mask": files_mask})
    df.to_csv(os.path.join(out_path, f"{prefix}_paths.csv"), index=None)


if __name__ == "__main__":
    main()
