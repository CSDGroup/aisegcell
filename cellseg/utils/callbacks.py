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
import ipdb
from pytorch_lightning.callsbacks import Callback


class CheckpointCallback(Callback):
    """
    If checkpoint is loaded run validation once to update best loss/best f1 scores for model saving.
    """

    def on_load_checkpoint(self, trainer):
        ipdb.set_trace()
