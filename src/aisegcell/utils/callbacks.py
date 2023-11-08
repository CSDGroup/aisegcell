#######################################################################################################################
# This script handels the loading and processing of the input dataset for cell segmentation with Unet                 #
# Contains the pytorch lightning DataModule                                                                           #
# Author:               Melinda Kondorosy, Daniel Schirmacher                                                         #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                               #
# Python Version:       3.8.7                                                                                         #
# PyTorch Version:      1.7.1                                                                                         #
# PyTorch Lightning Version: 1.5.9                                                                                    #
#######################################################################################################################
import torch
from pytorch_lightning.callbacks import Callback


class CheckpointCallback(Callback):
    """
    If checkpoint is loaded run validation once to update best loss/best f1 scores for model saving.
    """

    def __init__(self, retrain: bool = False):
        super().__init__()

        self.retrain = retrain

    def on_fit_start(self, trainer, pl_module):
        # get callback ids of loss_val and f1
        cb_ids = [
            hasattr(cb, "monitor")
            and cb.monitor in ("f1", "loss_val")
            and cb.best_model_score is not None
            for cb in trainer.callbacks
        ]
        cb_ids = [i for i, val in enumerate(cb_ids) if val]

        # update best_model_score if we are retraining
        if any(cb_ids):
            if self.retrain:
                for i in cb_ids:
                    # we assume the user has tested pretrained model and it is not sufficient
                    # --> we do not require a baseline for the untrained model
                    if trainer.callbacks[i].monitor == "f1":
                        trainer.callbacks[i].best_model_score = torch.tensor(
                            0.0
                        )
                        trainer.callbacks[i].best_k_models[
                            trainer.callbacks[i].best_model_path
                        ] = torch.tensor(0.0)
                    else:
                        trainer.callbacks[i].best_model_score = torch.tensor(
                            10.0
                        )
                        trainer.callbacks[i].best_k_models[
                            trainer.callbacks[i].best_model_path
                        ] = torch.tensor(10.0)
