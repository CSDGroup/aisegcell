import math

import pytest
import torch

from aisegcell.models.unet import LitUnet, UNet_rec


def test_Unet_rec():
    """
    Tests expected output size of an image by the Unet model

    """

    image = torch.rand((2, 1, 512, 512))

    unet = UNet_rec(bilinear=True, base_filters=32, receptive_field=128)

    image_hat = unet(image)

    assert image_hat.size() == image.size()


def test_Unet_sigmoid():
    """
    Testing range of expected output tensor.
    Uses Sigmoid function and threfore output tensor should be between [0,1]

    """

    input_image = torch.rand(1, 32, 12, 12)
    unet = UNet_rec(bilinear=True, base_filters=32, receptive_field=128)
    output = unet.outc(input_image)

    # convert output to numpy and make a iterable object out of it
    output = output.detach().numpy()
    listed_output = output[0][0][0][:]

    assert all(i >= 0 and i <= 1 for i in listed_output)


def test_Unet_rec_blocks():
    """
    Asserting number of blocks. According to the receptive field

    """
    receptive_field = 128
    unet = UNet_rec(bilinear=True, base_filters=32, receptive_field=128)

    assert unet.n_blocks == int(math.log2(receptive_field))


def test_LitUnet():
    """
    Throws error if receptive field isn't a power of two

    """

    with pytest.raises(AssertionError):
        LitUnet(
            n_classes=1,
            n_channels=1,
            bilinear=True,
            base_filters=32,
            receptive_field=7,
        )
