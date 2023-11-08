import os

import pandas as pd
import pytest

from aisegcell.preprocessing.generate_list import _generate_df


@pytest.fixture(scope="session")
def file_dir(tmp_path_factory):
    """
    Create temporary csv input file for Dataset
    """
    my_dir = tmp_path_factory.mktemp("data")
    my_file = my_dir / "a.txt"
    my_file.touch()
    my_file = my_dir / "1.txt"
    my_file.touch()
    my_file = my_dir / "e.txt"
    my_file.touch()

    return my_dir.as_posix()


def test_generate_df(file_dir):
    """
    Test sorted dataframe generation.
    """
    file_pattern = os.path.join(file_dir, "*.txt")
    df_pred = _generate_df(file_pattern, file_pattern)

    paths = [
        os.path.join(file_dir, "1.txt"),
        os.path.join(file_dir, "a.txt"),
        os.path.join(file_dir, "e.txt"),
    ]

    df_gt = pd.DataFrame(
        {
            "bf": paths,
            "mask": paths,
        }
    )

    assert df_pred.equals(df_gt)
