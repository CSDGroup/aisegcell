#######################################################################################################################
# This script contains the Unet model implementation with the Pytorch Lightning Module                                #
# Author:               Melinda Kondorosy, Daniel Schirmacher                                                         #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.8.7                                                                                         #
# PyTorch Version:      1.7.1                                                                                         #
# PyTorch Lightning Version: 1.5.9                                                                                    #
#######################################################################################################################
import math
import os
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from cellseg.utils.datamodule import save_image_mod
from cellseg.utils.evaluation import compute_f1, compute_iou

slope = 1e-2


def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(
            m.weight, a=slope, mode="fan_in", nonlinearity="leaky_relu"
        )


class DoubleConv(nn.Module):
    """
    This class does (convolution => [BN] => ReLU) * 2.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Constructor.


        Parameter
        ---------

        in_channels: int
            Input filters.

        out_channels: int
            Output filters.

        mid_channels: int
            intermediate filters.


        Return
        ------

        -
        """
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.double_conv.apply(init_weights)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    This class does downscaling with maxpool and then uses DoubleConv().
    """

    def __init__(self, in_channels, out_channels):
        """
        Constructor.


        Parameter
        ---------

        in_channels: int
            Input filters.

        out_channels: int
            Output filters.


        Return
        ------

        -
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    This class does upscaling and DoubleConv().
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        Constructor


        Parameter
        ---------

        in_channels: int
            Input filters.

        out_channels: int
            Output filters.

        bilinear: boolean
            Upsampling mode.


        Return
        ------

        -
        """
        super().__init__()

        # if bilinear use normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    This class is the output convolutional layer.
    """

    def __init__(self, in_channels, out_channels):
        """
        Constructor


        Parameter
        ---------

        in_channels: int
            Input filters.

        out_channels: int
            Output filters.


        Return
        ------

        -
        """
        super(OutConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.Sigmoid()
        )

        self.conv.apply(init_weights)
        nn.init.constant_(self.conv[0].bias, 0.0)

    def forward(self, x):
        return self.conv(x)


class UNet_rec(nn.Module):
    """
    This class contains the 2D UNet according to Ronneberger et al (2015).
    """

    def __init__(
        self,
        bilinear=True,
        base_filters=32,
        receptive_field=128,
        **kwargs,
    ):
        """
        Constructor of the UNet class. Following https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862


        Parameter
        ---------

        bilinear: bool
            Boolean flag indicating if upsampling interpolation should use bilinear mode.

        base_filter: int
            Number of convolutional filters doubled in the first layer.

        receptive_field: int
            Receptive field size of the innermost layer.


        Return
        ------

        -
        """

        super(UNet_rec, self).__init__()

        n_channels = 1  # number of channels and classes are fixed to 1
        n_classes = 1

        self.bilinear = bilinear
        self.base_filters = base_filters
        self.model_class = "unet"
        self.max_filters = 512
        self.receptive_field = receptive_field
        self.enc_block = []
        self.dec_block = []
        self.n_blocks = int(math.log2(self.receptive_field))  # number of layers
        factor = 2 if bilinear else 1

        # set up encoder
        self.inc = DoubleConv(n_channels, base_filters)
        for i in range(self.n_blocks):
            in_filters = base_filters * (2**i)
            in_filters = (
                in_filters
                if in_filters <= self.max_filters // 2
                else self.max_filters // 2
            )
            out_filters = base_filters * (2 ** (i + 1))
            out_filters = (
                out_filters
                if out_filters <= self.max_filters // 2
                else self.max_filters // 2
            )
            out_filters = self.max_filters if i == (self.n_blocks - 1) else out_filters
            out_filters = (
                out_filters // factor if i == (self.n_blocks - 1) else out_filters
            )

            self.enc_block.append(Down(in_filters, out_filters))

        # set up decoder
        for i in range(self.n_blocks - 1, -1, -1):
            in_filters = base_filters * (2 ** (i + 1))
            in_filters = (
                in_filters if in_filters <= self.max_filters else self.max_filters
            )
            out_filters = base_filters * (2**i)
            out_filters = (
                out_filters // factor
                if out_filters <= self.max_filters
                else self.max_filters // factor
            )
            out_filters = out_filters * factor if i == 0 else out_filters

            self.dec_block.append(Up(in_filters, out_filters, bilinear))

        self.outc = OutConv(base_filters, n_classes)
        self.dilation = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # add the list of modules to the current module
        self.enc_block = nn.ModuleList(self.enc_block)
        self.dec_block = nn.ModuleList(self.dec_block)

    def forward(self, x, dilate=False):
        encoder_output = []

        # encoder pathway
        x = self.inc(x)
        encoder_output.append(x)
        for module in self.enc_block:
            x = module(x)
            encoder_output.append(x)

        # decoder pathway
        for i, module in enumerate(self.dec_block):
            x = module(x, encoder_output[-(i + 2)])

        probs = self.outc(x)

        if dilate:
            probs = self.dilation(probs)

        return probs


class LitUnet(pl.LightningModule):
    """
    Set up of the LightningModule for the Unet model
    """

    def __init__(
        self,
        bilinear: bool = True,
        base_filters: int = 32,
        receptive_field: int = 128,
        learning_rate: float = 1e-3,
        loss_weight: float = 1.0,
        **kwargs,
    ):
        """


        Parameters
        ----------
        bilinear : bool
            If true use bilinear othterwise ConvTranspose.
        base_filters : int
            The default is 32. Number of conv. filters.
        receptive_field : int
            The default is 128. Must be power of 2.
        learning_rate : float, optional
            The default is 1e-3.

        Returns
        -------
        None.

        """
        super(LitUnet, self).__init__()

        # save hyperparameters from __init__ upon checkpoints in hparams.yaml
        self.save_hyperparameters()

        # assert input formats
        assert isinstance(
            base_filters, int
        ), f'base_filters is expected to be of type "int" but is of type "{type(base_filters)}".'
        assert isinstance(
            bilinear, bool
        ), f'bilinear is expected to be of type "bool" but is of type "{type(bilinear)}".'
        assert isinstance(
            receptive_field, int
        ), f'receptive_field is expected to be of type "int" but is of type "{type(receptive_field)}".'
        assert (
            math.log2(receptive_field) % 1 == 0
        ), "receptive_field is expected to be a power of 2."
        assert isinstance(
            learning_rate, float
        ), f'learning_rate is expected to be of type "float" but is of type "{type(learning_rate)}".'
        assert isinstance(
            loss_weight, float
        ), f'loss_weight is expected to be of type "float" but is of type "{type(loss_weight)}".'

        self.bilinear = bilinear
        self.base_filters = base_filters
        self.t_min = 0.1
        self.t_max = 0.6
        self.model_class = "unet"
        self.max_filters = 512
        self.receptive_field = receptive_field
        self.lr = learning_rate
        self.loss_weight = loss_weight
        self.unet = UNet_rec(
            bilinear=self.bilinear,
            base_filters=self.base_filters,
            receptive_field=self.receptive_field,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)

    def on_save_checkpoint(self, checkpoint) -> None:
        # save input variables which are not in the __init__function on checkpoints
        checkpoint["t_max"] = self.t_max
        checkpoint["t_min"] = self.t_min
        checkpoint["model_class"] = self.model_class
        checkpoint["max_filters"] = self.max_filters

    def on_load_checkpoint(self, checkpoint) -> None:
        self.t_max = checkpoint["t_max"]
        self.t_min = checkpoint["t_min"]
        self.model_class = checkpoint["model_class"]
        self.max_filters = checkpoint["max_filters"]

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        imgs, masks, ids = batch
        masks_hat = self(imgs)
        weight = (self.loss_weight - 1) * masks + 1

        loss = F.binary_cross_entropy(masks_hat, masks, weight=weight)
        self.log("loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        return {"loss": loss, "masks_hat": masks_hat, "masks": masks}

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        imgs, masks, ids = batch
        masks_hat = self(imgs)
        weight = (self.loss_weight - 1) * masks + 1

        loss_val = F.binary_cross_entropy(masks_hat, masks, weight=weight)
        self.log("loss_val", loss_val, on_step=True, on_epoch=True, sync_dist=True)

        f1_scores = compute_f1(masks_hat, masks, self.t_min, self.t_max)
        self.log(
            "f1",
            f1_scores["f1"].mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # compute and log IOU
        iou_all = compute_iou(masks_hat, masks)
        iou = iou_all["iou"]
        iou_big = iou_all["iou_big"]
        iou_small = iou_all["iou_small"]

        self.log(
            "iou",
            iou.mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "iou_big",
            iou_big.mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "iou_small",
            iou_small.mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss_val

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        imgs, masks, ids = batch
        masks_hat = self(imgs)
        weight = (self.loss_weight - 1) * masks + 1

        # save loss
        loss_test = F.binary_cross_entropy(masks_hat, masks, weight=weight)
        self.log("loss_test", loss_test, on_step=True, on_epoch=True, sync_dist=False)

        # save f1 and associated metrics
        f1_scores = compute_f1(masks_hat, masks, self.t_min, self.t_max)
        self.log("f1", f1_scores["f1"][0], on_step=True, on_epoch=True, sync_dist=False)
        self.log("tp", f1_scores["tp"][0], on_step=True, on_epoch=True, sync_dist=False)
        self.log("fp", f1_scores["fp"][0], on_step=True, on_epoch=True, sync_dist=False)
        self.log("fn", f1_scores["fn"][0], on_step=True, on_epoch=True, sync_dist=False)
        self.log(
            "splits",
            f1_scores["splits"][0],
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "merges",
            f1_scores["merges"][0],
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "inaccurate_masks",
            f1_scores["inaccurate_masks"][0],
            on_step=True,
            on_epoch=True,
            sync_dist=False,
        )

        iou_all = compute_iou(masks_hat, masks)
        iou = iou_all["iou"]
        iou_big = iou_all["iou_big"]
        iou_small = iou_all["iou_small"]

        self.log(
            "iou",
            iou.mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "iou_big",
            iou_big.mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.log(
            "iou_small",
            iou_small.mean(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # binarise inferred masks for visualisation
        # (cells = 1, background = 0)
        masks_hat[masks_hat < 0.5] = 0
        masks_hat[masks_hat >= 0.5] = 1

        # save inferred masks
        for mask, i in zip(masks_hat, ids):
            mask_path = self.trainer.datamodule.data_test.data.iloc[i.item(), 0].split(
                os.sep
            )[-1]
            mask_path = os.path.join(
                self.trainer.logger.log_dir, "test_masks", mask_path
            )
            save_image_mod(mask, mask_path, nrow=1, padding=0)

        return loss_test

    def predict_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        imgs, ids = batch
        masks_hat = self(imgs)

        masks_hat[masks_hat < 0.5] = 0
        masks_hat[masks_hat >= 0.5] = 1

        # save predicted masks
        for img, i in zip(masks_hat, ids):
            img_path = self.trainer.datamodule.data_predict.data.iloc[
                i.item(), 0
            ].split(os.sep)[-1]
            img_path = os.path.join(
                self.trainer.logger.log_dir, "predicted_masks", img_path
            )
            save_image_mod(img, img_path, nrow=1, padding=0)

        return masks_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def on_test_start(self):
        os.makedirs(
            os.path.join(self.trainer.logger.log_dir, "test_masks"), exist_ok=True
        )

    def on_predict_start(self):
        os.makedirs(
            os.path.join(self.trainer.logger.log_dir, "predicted_masks"), exist_ok=True
        )
