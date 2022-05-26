import numpy as np
import pandas as pd
import pytest
import torch
import torchvision.transforms.functional as tf
from torchvision import transforms

from cellseg.utils.datamodule import Dataset, Dataset_predict, Dataset_test


@pytest.fixture(scope="session")
def csv_file(tmp_path_factory):
    """
    Create temporary csv input file for Dataset
    """
    my_file = tmp_path_factory.mktemp("data") / "myfile.csv"
    my_file.touch()

    df = pd.DataFrame({"a": ["a", "b", "c"]})
    df.to_csv(my_file.as_posix(), index=False)

    return my_file.as_posix()


def test_init(csv_file):
    """
    throws error after wrong shape insertion
    Shape should be of type integers
    """

    with pytest.raises(AssertionError):
        _ = Dataset(csv_file, shape=(4.5, 8.2))


def test_preprocess(csv_file):
    """
    tests preprocess step. Input image and mask
    -->output should be normalised tensor
    """
    dataset = Dataset(csv_file, shape=(2, 2))

    image = np.array([[0, 51], [255, 0]])
    mask = np.array([[51, 102], [0, 255]])

    image_tensor = dataset._preprocess(image, mask)
    output_image, output_mask = image_tensor

    expected_image = torch.Tensor([[0.0, 0.2], [1.0, 0.0]]).type(torch.FloatTensor)
    expected_mask = torch.Tensor([[0.2, 0.4], [0, 1.0]]).type(torch.FloatTensor)

    assert (
        torch.all(torch.eq(output_image, expected_image)).item()
        and torch.all(torch.eq(output_mask, expected_mask)).item()
    )


def test_preprocess_transform(csv_file):
    """
    Tests transformation in preprocess step.
    Tests horizontal flip (hflip) on image and mask
    """

    transform = transforms.Compose(
        [transforms.Normalize(0.0, 255.0), transforms.Lambda(lambda img: tf.hflip(img))]
    )
    dataset = Dataset(csv_file, transform, shape=(2, 2))

    image = np.array([[0, 51], [255, 0]])
    mask = np.array([[51, 102], [0, 255]])

    image_tensor = dataset._preprocess(image, mask)
    output_image, output_mask = image_tensor

    expected_image = torch.Tensor([[0.2, 0.0], [0.0, 1.0]]).type(torch.FloatTensor)
    expected_mask = torch.Tensor([[0.4, 0.2], [1.0, 0.0]]).type(torch.FloatTensor)

    assert (
        torch.all(torch.eq(output_image, expected_image)).item()
        and torch.all(torch.eq(output_mask, expected_mask)).item()
    )


def test_preprocess_test(csv_file):
    """
    Tests the predict Dataset
    tests normalisation step

    """
    dataset_test = Dataset_test(csv_file)

    image = np.array([[0, 51], [255, 0]])
    mask = np.array([[51, 102], [0, 255]])

    image_tensor = dataset_test._preprocess(image, mask)
    output_image, output_mask = image_tensor

    expected_image = torch.Tensor([[0.0, 0.2], [1.0, 0.0]]).type(torch.FloatTensor)
    expected_mask = torch.Tensor([[0.2, 0.4], [0, 1.0]]).type(torch.FloatTensor)

    assert (
        torch.all(torch.eq(output_image, expected_image)).item()
        and torch.all(torch.eq(output_mask, expected_mask)).item()
    )


def test_preprocess_predict(csv_file):
    """
    Tests the predict Dataset
    tests normalisation step

    """
    dataset_pred = Dataset_predict(csv_file)

    image = np.array([[0, 51], [255, 0]])

    output_image = dataset_pred._preprocess(image)

    expected_image = torch.Tensor([[0.0, 0.2], [1.0, 0.0]]).type(torch.FloatTensor)

    assert torch.all(torch.eq(output_image, expected_image))
