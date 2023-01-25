# -*- coding: utf-8 -*-
#######################################################################################################################
# This script handels the loading and processing of the input dataset for cell segmentation with Unet                 #
# Contains the pytorch lightning DataModule                                                                           #
# Author:               Melinda Kondorosy, Daniel Schirmacher                                                         #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.8.7                                                                                         #
# PyTorch Version:      1.7.1                                                                                         #
# PyTorch Lightning Version: 1.5.9                                                                                    #
#######################################################################################################################

import pathlib
import random
from os import path
from typing import BinaryIO, List, Optional, Text, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from skimage import io
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid


class Dataset:
    """
    Handling of data.
    Takes image and mask and transforms them in preprocess()
    (adding of color channel and crop and padding of image and mask).
    Returns image, mask and index of image in getitem()

    """

    def __init__(
        self,
        path_data: str,
        transform_spatial: Optional[transforms.transforms.Compose] = None,
        transform_intensity: Optional[transforms.transforms.Compose] = None,
        shape: Tuple[int, int] = (512, 512),
    ):
        """


        Parameters
        ----------
        path_data : str
            path to csv file with images and masks.
        transform_spatial : Optional[transforms.transforms.Compose], optional
            transformation which are applied on image and mask
        transform_intensity : Optional[transforms.transforms.Compose], optional
            transformation which are applied to image only
        shape : Tuple[int, int], optional
            height and width which all images and masks will have in the end. The default is (512, 512).

        Returns
        -------
        None.

        """

        super().__init__()

        # assert input path
        assert (
            type(path_data) == str
        ), f'path_data should be of type "str" but is of type "{type(path_data)}".'

        assert path.exists(
            path_data
        ), f'path_data does not exist, you typed: "{path_data}".'

        # transformation
        if transform_spatial is None:
            transform_spatial = transforms.Compose([transforms.Normalize(0.0, 255.0)])
        else:
            assert (
                type(transform_spatial) == transforms.transforms.Compose
            ), f'transform_spatial should be of type "torchvision.transforms.transforms.Compose" but is of type "{type(transform_spatial)}".'

        if transform_intensity is not None:
            assert (
                type(transform_intensity) == transforms.transforms.Compose
            ), f'transform_intensity should be of type "torchvision.transforms.transforms.Compose" but is of type "{type(transform_intensity)}".'

        # assert shape
        assert (
            type(shape) == tuple
        ), f'type of shape should be tuple instead it is of type: "{type(shape)}".'

        assert all(
            isinstance(i, int) for i in shape
        ), "values of shape should be of type integer."

        self.path_data = path_data
        self.data = pd.read_csv(path_data)
        self.shape = shape
        self.transform_spatial = transform_spatial
        self.transform_intensity = transform_intensity
        self._padder = transforms.RandomCrop(self.shape, pad_if_needed=True)

    def __len__(self):
        """

        Returns
        -------
        returns length of dataset

        """
        return len(self.data)

    def _preprocess(self, image: np.array, mask: np.array) -> torch.Tensor:
        """
        Normalise, augment and transform image and mask


        Parameters
        ----------
        image : np.array
            input images (height, width).
        mask : np.array
            input masks (height, width).

        Returns
        -------
        image_trans : torch.tensor
            transformed image (channel, height, width).
        mask_trans : torch.tensor
            transformed mask (channel, height, width).

        """

        # add color channel # NOTE: images and masks are expected to be grayscale
        assert (
            len(image.shape) == 2
        ), f'images are expected to be grayscale and len(image.shape)==2, here it is: "{len(image.shape)}".'
        assert (
            len(mask.shape) == 2
        ), f'masks are expected to be grayscale and len(mask.shape)==2, here it is: "{len(mask.shape)}".'

        image = np.expand_dims(image, axis=2)
        image = image.transpose((2, 0, 1))  # (colorchannel,height,width)

        mask = np.expand_dims(mask, axis=2)
        mask = mask.transpose((2, 0, 1))  # (colorchannel,height,width)

        image_trans = torch.from_numpy(image).type(torch.FloatTensor)
        mask_trans = torch.from_numpy(mask).type(torch.FloatTensor)

        # merge image and masks for padding and transformations
        # to apply the same random augmentations to images and masks
        _, height, width = image_trans.size()
        merged = torch.zeros(2, height, width)
        merged[0, :, :] = image_trans
        merged[1, :, :] = mask_trans

        merged_trans = self._padder(merged)
        merged_trans = self.transform_spatial(merged_trans)

        # split images and masks after transformation again
        image_trans = merged_trans[0, :, :]
        image_trans = image_trans[None, :, :]

        mask_trans = merged_trans[1, :, :]
        mask_trans = mask_trans[None, :, :]

        # apply self.transfom_intensity
        if self.transform_intensity is not None:
            image_trans = self.transform_intensity(image_trans)

        return image_trans, mask_trans

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        read data (csv file)

        Parameters
        ----------
        idx : int
            index to image.

        Returns
        -------
        image : torch.tensor
            preprocessed image.
        mask : torch.tensor
            preprocessed mask.
        idx : int
            index to image.

        """

        image_path = self.data.iloc[idx, 0]
        image = io.imread(image_path)
        # image in format (height, width) or (depth, height, width)

        mask_path = self.data.iloc[idx, 1]
        mask = io.imread(mask_path)
        # mask in format (height, width) or (depth, height, width)

        # assert all masks are semantic segmentation in 8bit format
        mask[mask > 0] = 255
        mask = mask.astype(np.uint8)

        # preprocess image and mask
        image, mask = self._preprocess(image, mask)

        return image, mask, idx

    def display_image(self, idx: Union[int, str]):
        """
        Displays image and mask as matplotlib plot.

        Parameters
        ----------
        idx : Union[int, str]
            Index to image, eitcher "random" or int.

        Raises
        ------
        Warning
            display image is only availabe for 2D images.

        Returns
        -------
        None.

        """

        assert (
            type(idx) == random or type(idx) == int
        ), f'idx must be either a int or "random". Here it is {type(idx)}'

        if idx == "random":
            idx = random.choice(self.data.index)

        image = io.imread(self.data.iloc[idx, 0])

        mask = io.imread(self.data.iloc[idx, 1])

        if len(image.shape) == 2:
            f = plt.figure()
            f.add_subplot(1, 2, 1)
            plt.imshow(image, cmap="gray", vmin=0, vmax=255)
            f.add_subplot(1, 2, 2)
            plt.imshow(mask, cmap="gray", vmin=0, vmax=255)
            plt.show(block=True)
        else:
            raise Warning("display_image() is only available for 2D images.")


class Dataset_test:
    """
    Handling of test data.
    Takes image and mask and normalises them.
    Returns image, mask and index of image in getitem()

    """

    def __init__(
        self,
        path_data_test: str,
    ):
        """


        Parameters
        ----------
        path_data_test : str
            path to csv file with images and masks.

        Returns
        -------
        None.

        """

        super().__init__()

        # assert input path
        assert (
            type(path_data_test) == str
        ), f'path_data_test should be of type "str" but is of type "{type(path_data_test)}".'

        assert path.exists(
            path_data_test
        ), f'path_data does not exist, you typed: "{path_data_test}".'

        self.path_data_test = path_data_test
        self.data = pd.read_csv(path_data_test)

    def __len__(self):
        """

        Returns
        -------
        returns length of dataset

        """
        return len(self.data)

    def _preprocess(self, image: np.array, mask: np.array) -> torch.Tensor:
        """
        Normalise and transform image and mask


        Parameters
        ----------
        image : np.array
            input images (height, width).
        mask : np.array
            input masks (height, width).

        Returns
        -------
        image_trans : torch.tensor
            transformed image (channel, height, width).
        mask_trans : torch.tensor
            transformed mask (channel, height, width).

        """

        # add color channel # NOTE: images and masks are expected to be grayscale
        assert (
            len(image.shape) == 2
        ), f'images are expected to be grayscale and len(image.shape)==2, here it is: "{len(image.shape)}".'
        assert (
            len(mask.shape) == 2
        ), f'masks are expected to be grayscale and len(mask.shape)==2, here it is: "{len(mask.shape)}".'

        image = np.expand_dims(image, axis=2)
        image = image.transpose((2, 0, 1))  # (colorchannel,height,width)

        mask = np.expand_dims(mask, axis=2)
        mask = mask.transpose((2, 0, 1))  # (colorchannel,height,width)

        image_trans = torch.from_numpy(image).type(torch.FloatTensor)
        mask_trans = torch.from_numpy(mask).type(torch.FloatTensor)

        # normalize image and mask
        transform = transforms.Compose([transforms.Normalize(0.0, 255.0)])
        image_trans = transform(image_trans)
        mask_trans = transform(mask_trans)

        return image_trans, mask_trans

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        read data (csv file)

        Parameters
        ----------
        idx : int
            index to image.

        Returns
        -------
        image : torch.tensor
            preprocessed image.
        mask : torch.tensor
            preprocessed mask.
        idx : int
            index to image.

        """

        image_path = self.data.iloc[idx, 0]
        image = io.imread(image_path)
        # image in format (height, width) or (depth, height, width)

        mask_path = self.data.iloc[idx, 1]
        mask = io.imread(mask_path)
        # mask in format (height, width) or (depth, height, width)

        # assert all masks are semantic segmentation in 8bit format
        mask[mask > 0] = 255
        mask = mask.astype(np.uint8)

        # preprocess image and mask
        image, mask = self._preprocess(image, mask)

        return image, mask, idx


class Dataset_predict:
    """
    Handling of prediction data.
    Takes images and normalises them.
    Returns image and index of image in getitem()

    """

    def __init__(
        self,
        path_data_predict: str,
    ):
        """

        Parameters
        ----------
        path_data_predict : str
            path to prediction data

        Returns
        -------
        None.

        """

        super().__init__()

        # assert input path
        assert path.exists(
            path_data_predict
        ), f'path to path_predict_data does not exist, you typed: "{path_data_predict}".'

        assert (
            type(path_data_predict) == str
        ), f'path_data_predict should be of type "str" but is of type "{type(path_data_predict)}".'

        self.path_data_predict = path_data_predict
        self.data = pd.read_csv(path_data_predict)

    def __len__(self):
        """

        Returns
        -------
        returns length of predict dataset

        """
        return len(self.data)

    def _preprocess(self, image: np.array) -> torch.Tensor:
        """
        transform image


        Parameters
        ----------
        image : np.array
            input image of shape (height, width).

        Returns
        -------
        image_trans : TYPE
            transformed image of shape (channel, height, width).

        """

        # add color channel # NOTE: image is expected to be grayscale
        assert (
            len(image.shape) == 2
        ), f'images are expected to be grayscale and len(image.shape)==2, here it is: "{len(image.shape)}".'

        image = np.expand_dims(image, axis=2)
        image = image.transpose((2, 0, 1))  # (colorchanel,height,width)

        image_trans = torch.from_numpy(image).type(torch.FloatTensor)

        transform = transforms.Normalize(0.0, 255.0)
        image_trans = transform(image_trans)

        return image_trans

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """
        read data (csv file)

        Parameters
        ----------
        idx : int
            index to image.

        Returns
        -------
        image : torch.Tensor
            preprocessed image (channel, height, width).
        idx : torch.Tensor
            index to image.

        """

        image_path = self.data.iloc[idx, 0]
        image = io.imread(image_path)
        # image in format (height, width) or (depth, height, width)

        # preprocess image and mask
        image = self._preprocess(image)

        return image, idx


class DataModule(pl.LightningDataModule):
    """
    Pytorch lightning class which encapsulates the data preprocess and loading steps
    """

    def __init__(
        self,
        path_data: str,
        path_data_val: str,
        path_data_test: Optional[str] = None,
        path_data_predict: Optional[str] = None,
        batch_size: int = 2,
        shape: Tuple[int, int] = (512, 512),
        transform_intensity: bool = False,
    ):
        """


        Parameters
        ----------
        path_data : str
            path to train data (csv file).
        path_data_val : str
            path to validation data (csv file).
        path_data_test : Optional[str], optional
            path to test data (csv file). The default is None.
        path_data_predict : Optional[str], optional
            path to prediction data (csv file). The default is None.
        batch_size : int, optional
            The default is 2.
        shape : Tuple[int, int], optional
            The default is (512, 512).
        transform_intensity : bool, optional
            If True random intensity transformations will be applied to image.

        Returns
        -------
        None.

        """

        super().__init__()

        self.path_data = path_data
        self.path_data_val = path_data_val
        self.path_data_test = path_data_test
        self.path_data_predict = path_data_predict
        self.batch_size = batch_size
        self.shape = shape
        self.transform_intensity = transform_intensity

    def setup(self, stage: Optional[str] = None):
        """
        Instantiate datasets

        """

        if stage == "fit" or stage is None:
            transform_spatial = transforms.Compose(
                [
                    transforms.Normalize(0.0, 255.0),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(45),
                ]
            )

            if self.transform_intensity:
                transform_intensity = transforms.Compose(
                    [
                        transforms.ColorJitter(
                            brightness=0.7, contrast=0.5, saturation=0.5, hue=0.5
                        ),
                        transforms.GaussianBlur(kernel_size=5),
                        transforms.RandomAdjustSharpness(4, p=0.5),
                    ]
                )
            else:
                transform_intensity = None

            self.data = Dataset(
                self.path_data,
                transform_spatial=transform_spatial,
                transform_intensity=transform_intensity,
                shape=self.shape,
            )
            self.data_val = Dataset(
                self.path_data_val,
                transform_spatial=transform_spatial,
                transform_intensity=transform_intensity,
                shape=self.shape,
            )

        if stage == "test" or stage is None:
            if self.path_data_test is not None:
                self.data_test = Dataset_test(
                    self.path_data_test,
                )
            else:
                raise ValueError("path_data_test is missing")

        if stage == "predict" or stage is None:
            if self.path_data_predict is not None:
                self.data_predict = Dataset_predict(
                    self.path_data_predict,
                )
            else:
                raise ValueError("path_data_predict is missing")

    def train_dataloader(self):
        return DataLoader(
            self.data, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.data_predict, batch_size=self.batch_size, num_workers=4)


def save_image_mod(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    format: Optional[str] = None,
) -> None:
    """
    torchvision.utils.save_image modified to save gray_scale images.


    Parameter
    ---------

        tensor: Tensor or list
            Image to be saved. If given a mini-batch tensor, saves the tensor as a grid of images by
            calling ``make_grid``.

        fp: string or file object
            A filename or a file object.

        format(Optional):
            If omitted, the format to use is determined from the filename extension. If a file object was used
            instead of a filename, this parameter should always be used.

        **kwargs: Other arguments are documented in ``make_grid``.


    Return
    ------

    -
    """
    if len(tensor.size()) == 3:
        n_channels = tensor.size()[0]
    elif len(tensor.size()) == 4:
        n_channels = tensor.size()[1]
    else:
        raise TypeError(
            f"Tensor.size() is expected to be of len 3 or 4, but is of len {len(tensor.size())}."
        )

    grid = make_grid(
        tensor,
        nrow=nrow,
        padding=padding,
        pad_value=pad_value,
        normalize=normalize,
        range=range,
        scale_each=scale_each,
    )

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer --> modified to obtain grayscale image
    if n_channels == 1:
        ndarr = grid[0].mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    else:
        ndarr = (
            grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )

    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
