########################################################################################################################
# This script constructs a .csv file containing matching bf/mask paths in each row.                                    #
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
        required=True,
        help="Path (glob pattern, e.g. 'C:/path/to/images/*.png') to bright field images. Naming convention must match "
        "naming convention of --mask s.t. alphanumerically sorted paths are matching.",
    )

    parser.add_argument(
        "--mask",
        type=str,
        required=True,
        help="Path (glob pattern, e.g. 'C:/path/to/masks/*.png') to segmentation masks.",
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to output directory.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="train",
        help="Prefix for output file name (i.e. '{prefix}_paths.csv'). Default is 'train'.",
    )

    return parser.parse_args()


def _generate_df(path_bf: str, path_mask: str) -> pd.DataFrame:
    files_bf = glob.glob(path_bf)
    files_bf.sort()
    files_mask = glob.glob(path_mask)
    files_mask.sort()

    df = pd.DataFrame({"bf": files_bf, "mask": files_mask})

    return df


def main():
    args = arg_parse()

    path_bf = args.bf
    path_mask = args.mask
    prefix = args.prefix
    path_out = args.out

    df = _generate_df(path_bf, path_mask)
    os.makedirs(path_out, exist_ok=True)
    df.to_csv(os.path.join(path_out, f"{prefix}_paths.csv"), index=None)


if __name__ == "__main__":
    main()
