[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Tests](https://github.com/CSDGroup/cell_segmentation/workflows/tests/badge.svg)](https://github.com/CSDGroup/cell_segmentation/actions)
[![codecov](https://codecov.io/gh/CSDGroup/cell_segmentation/branch/main/graph/badge.svg?token=63T8R6MUMB)](https://codecov.io/gh/CSDGroup/cell_segmentation)

# *CellSeg* - Overview
This repository contains a `torch` implementation of U-Net ([Ronneberger et al., 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)).
We provide [trained](#trained-models) models for segmenting nuclei and whole cells in bright field images.
Please cite [this paper](#citation) if you are using this code in your research.

## Contents
  - [Installation](#installation)
  - [Training](#training)
    - [Trained models](#trained-models)
  - [Testing](#testing)
  - [Predicting](#predicting)
    - [napari plugin](#napari-plugin)
  - [image annotation tools](#image-annotation-tools)
  - [Troubleshooting & support](#troubleshooting-&-support)
  - [Citation](#citation)

## Installation
Installation requires a command line application (e.g. `Terminal`) with [`git` installed](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
If you operate on `Windows` we recommend using [`Ubuntu on Windows`](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview).
Alternatively, you can install [`Anaconda`](https://docs.anaconda.com/anaconda/user-guide/getting-started/) and
use `Anaconda Powershell Prompt`. An introductory tutorial on how to use `git` and GitHub can be found [here](https://www.w3schools.com/git/default.asp?remote=github).

1) We recommend using a [virtual environment](https://realpython.com/python-virtual-environments-a-primer/). [Here](https://testdriven.io/blog/python-environments/)
is a list of different python virtual environment tools. Open your command line application and create a 
(e.g. `conda`) virtual environment

    ```bash
    conda create -n cellseg python=3.8
    ```

2) Activate your virtual environment

    ```bash
    conda activate cellseg
    ```

3) (Optional) If you use `Anaconda Powershell Prompt`, install `git` through `conda`

    ```bash
    conda install -c anaconda git
    ```

4) clone the repository (consider `ssh` alternative)

    ```bash
    # change directory
    cd /path/to/directory/to/clone/repository/to

    git clone https://github.com/CSDGroup/cell_segmentation.git
    ```

5) Navigate to the cloned directory

    ```bash
    cd cell_segmentation
    ```

6) Install `cell_segmentation`
    ```bash
    # update pip
    pip install -U pip
    ```

    1) as a user

        ```bash
        pip install .
        ```
    2) as a developer (in editable mode with development dependencies and pre-commit hooks)
 
        ```bash
        pip install -e ".[dev]"
        pre-commit install
        ```

7) (Optional) `GPUs` greatly speed up training and inference of U-Net and are available for `torch` (`v1.10.2`) for 
`Windows` and `Linux`. Check if your `GPU(s)` are CUDA compatible ([`Windows`](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#verify-you-have-a-cuda-capable-gpu),
 [`Linux`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#verify-you-have-a-cuda-capable-gpu)) and 
 update their drivers if necessary.

8) [Install `torch`/`torchvision`](https://pytorch.org/get-started/previous-versions/) compatible with your system. `cell_segmentation` was
tested with `torch` version `1.10.2`, `torchvision` version `0.11.3`, and `cuda` version `11.3.1`. Depending on
your OS, your `CPU` or `GPU` (and `CUDA` version) the installation may change

```bash
# Windows/Linux CPU
pip install torch==1.10.2+cpu torchvision==0.11.3+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Windows/Linux GPU (CUDA 11.3.X)
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# macOS CPU
pip install torch==1.10.2 torchvision==0.11.3

```

9) [Install `pytorch-lightning`](https://www.pytorchlightning.ai). `cell_segmentation` was tested with version `1.5.9`.

```bash
# note the installation of v1.5.9 does not use pip install lightning
pip install pytorch-lightning==1.5.9
```


## Training
Training U-Net is as simple as calling the command `cellseg_train`. `cellseg_train` is available if you activate
the virtual environment you [installed](#installation) and can be called with the following arguments:

  - `--help`: show help message
  - `--data`: Path to CSV file containing training image file paths. The CSV file must have the columns `bf` and
    `mask`. 
  - `--data_val`: Path to CSV file containing validation image file paths (same format as `--data`).
  - `--output_base_dir`: Path to output directory.
  - `--model`: Model type to train (currently only U-Net). Default is "Unet".
  - `--checkpoint`: Path to checkpoint file matching `--model`. Only necessary if continuing a model training.
    Default is `None`.
  - `--devices`: Devices to use for model training. If you want to use GPU(s) you have to provide `int` IDs. 
    Multiple GPU IDs have to be listed separated by spacebar (e.g. `2 5 9`). If you want to use the CPU you have
    to use "cpu". Default is "cpu".
  - `--epochs`: Number of training epochs. Default is 5.
  - `--batch_size`: Number of samples per mini-batch. Default is 2.
  - `--lr`: Learning rate of the optimizer. Default is 1e-4.
  - `--base_filters`: Number of base_filters of Unet. Default is 32.
  - `--shape`: Shape [heigth, width] that all images will be cropped/padded to before model submission. Height
    and width cannot be smaller than `--receptive_field`. Default is [1024,1024].
  - `--receptive_field` Receptive field of a neuron in the deepest layer. Default is 128.
  - `--log_frequency`: Log performance metrics every N gradient steps during training. Default is 50.
  - `--loss_weight`: Weight of the foreground class compared to the background class for the binary cross entropy loss.
    Default is 1.
  - `--bilinear`: If flag is used, use bilinear upsampling, else transposed convolutions.
  - `--multiprocessing`: If flag is used, all GPUs given in devices will be used for traininig. Does not support CPU.
  - `--retrain`: If flag is used, best scores for model saving will be reset (required for training on new data).
  - `--transform_intensity`: If flag is used random intensity transformations will be applied to input images.
  - `--seed`: None or Int to use for random seeding. Default is `None`.

The command `cellseg_generate_list` can be used to write CSV files for `--data` and `--data_val` and 
has the following arguments:
  - `--help`: show help message
  - `--bf`: Path ([`glob`](https://docs.python.org/3/library/glob.html) pattern) to input images (e.g. bright field). Naming convention must match naming convention of `--mask`.
  - `--mask`: Path (`glob` pattern) to segmentation masks corresponding to `--bf`.
  - `--out`: Directory to which output file is saved.
  - `--prefix`: Prefix for output file name (i.e. `{PREFIX}_paths.csv`). Default is "train".

Use [wildcard characters](https://linuxhint.com/bash_wildcard_tutorial/) like `*` to select all files you want to
input to `--bf` and `--mask` (see example below).

Consider the following example:
```bash
# activate the virtual environment
conda activate cellseg

# generate CSV files for data and data_val
cellseg_generate_list \
  --bf "/path/to/train_images/*/*.png" # i.e. select all PNG files in all sub-directories of /path/to/train_images\
  --mask "/path/to/train_masks/*/*mask.png" # i.e. select all files in all sub-directories that end with "mask.png"\
  --out /path/to/output_directory \
  --prefix train

cellseg_generate_list \
  --bf "/path/to/val_images/*.png" \
  --mask "/path/to/val_masks/*.png" \
  --out /path/to/output_directory \
  --prefix val

# starting multi-GPU training
cellseg_train \
  --data /path/to/output_directory/train_paths.csv \
  --data_val /path/to/output_directory/val_paths.csv \
  --model Unet \
  --devices 2 4 # use GPU 2 and 4 \
  --output_base_dir /path/to/results/folder \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-3 \
  --base_filters 32 \
  --shape 1024 512 \
  --receptive_field 128 \
  --log_frequency 5 \
  --loss_weight 1 \
  --bilinear  \
  --multiprocessing # required if you use multiple --devices \
  --transform_intensity \
  --seed 123

# OR retrain an existing checkpoint with single GPU
cellseg_train \
  --data /path/to/output_directory/train_paths.csv \
  --data_val /path/to/output_directory/val_paths.csv \
  --model Unet \
  --checkpoint /path/to/checkpoint/file.ckpt
  --devices 0 \
  --output_base_dir /path/to/results/folder \
  --epochs 10 \
  --batch_size 8 \
  --lr 1e-3 \
  --base_filters 32 \
  --shape 1024 1024 \
  --receptive_field 128 \
  --log_frequency 5 \
  --loss_weight 1 \
  --bilinear  \
  --transform_intensity \
  --seed 123
```

The output of `cellseg_train` will be stored in subdirectories `{DATE}_Unet_{ID1}/lightning_logs/version_{ID2}/` at
`--output_base_dir`. Its contents are:

  - `hparams.yaml`: stores hyper-parameters of the model (used by `pytorch_lightning.LightningModule`)
  - `metrics.csv`: contains all metrics tracked during training
    - `loss_step`: training loss (binary cross-entropy) per gradient step
    - `epoch`: training epoch
    - `step`: training gradient step
    - `loss_val_step`: validation loss (binary cross-entropy) per validation mini-batch
    - `f1_step`: [f1 score](https://www.biorxiv.org/content/10.1101/803205v2) per validation mini-batch
    - `iou_step`: average of `iou_small_step` and `iou_big_step` per validation mini-batch
    - `iou_big_step`: [intersection over union](https://www.biorxiv.org/content/10.1101/803205v2) of objects with 
      \> 2000 px in size per validation mini-batch
    - `iou_small_step`: [intersection over union](https://www.biorxiv.org/content/10.1101/803205v2) of objects
      with <= 2000 px in size per validation mini-batch
    - `loss_val_epoch`: average `loss_val_step` over all validation steps per epoch
    - `f1_epoch`: average `f1_step` over all validation steps per epoch
    - `iou_epoch`: average `iou_step` over all validation steps per epoch
    - `iou_big_epoch`: average `iou_big_epoch` over all validation steps per epoch
    - `iou_small_epoch`: average `iou_small_epoch` over all validation steps per epoch
    - `loss_epoch`: average `loss_step` over all training gradient steps per epoch
  - `checkpoints`: model checkpoints are stored in this directory. Path to model checkpoints are used as input to
    `--checkpoint` of `cellseg_train` or `--model` of `cellseg_test` and `cellseg_predict`.
    - `best-f1-epoch={EPOCH}-step={STEP}.ckpt`: model weights with the (currently) highest `f1_epoch`
    - `best-iou-epoch={EPOCH}-step={STEP}.ckpt`: model weights with the (currently) highest `iou_epoch`
    - `best-loss-epoch={EPOCH}-step={STEP}.ckpt`: model weights with the (currently) lowest `loss_val_epoch`
    - `latest-epoch={EPOCH}-step={STEP}.ckpt`: model weights of the (currently) latest checkpoint

### Trained models
We provide trained models:

| modality | image format | example image | description | availability |
| :-- | :-: | :-: | :-: | :-- |
| nucleus segmentation | 2D grayscale | <img src="https://github.com/CSDGroup/cell_segmentation/raw/main/images/nucseg.png" title="example nucleus segmentation" width="180px" align="center"> | Trained on a data set (link to data set) of 9849 images (~620k nuclei). | link to model weights (link to zenodo/model zoo) |
| whole cell segmentation | 2D grayscale | <img src="https://github.com/CSDGroup/cell_segmentation/raw/main/images/cellseg.png" title="example whole cell segmentation" width="180px" align="center"> | Trained on a data set (link to data set) of 226 images (~12k cells). | link to model weights (link to zenodo/model zoo) |

## Testing
A trained U-Net can be tested with `cellseg_test`. `cellseg_test` returns predicted masks and performance
metrics. `cellseg_test` can be called with the following arguments:

  - `--help`: show help message
  - `--data`: Path to CSV file containing test image file paths. The CSV file must have the columns `bf` and
    `--mask`. 
  - `--model`: Path to checkpoint file of trained pytorch_lightning.LightningModule.
  - `--suffix`: Suffix to append to all mask file names.
  - `--output_base_dir`: Path to output directory.
  - `--devices`: Devices to use for model training. If you want to use GPU(s) you have to provide `int` IDs. 
    Multiple GPU IDs have to be listed separated by spacebar (e.g. `2 5 9`). If multiple GPUs are provided only
    the first ID will be used. If you want to use the CPU you have to use "cpu". Default is "cpu".

Make sure to activate the virtual environment created during [installation](#installation) before calling
`cellseg_test`.

Consider the following example:
```bash
# activate the virtual environment
conda activate cellseg

# generate CSV file for data
cellseg_generate_list \
  --bf "/path/to/test_images/*.png" \
  --mask "/path/to/test_masks/*.png" \
  --out /path/to/output_directory \
  --prefix test

# run testing
cellseg_test \
  --data /path/to/output_directory/test_paths.csv \
  --model /path/to/checkpoint/file.ckpt \
  --suffix mask \
  --output_base_dir /path/to/results/folder \
  --devices 0 # predict with GPU 0
```

The output of `cellseg_test` will be stored in subdirectories `lightning_logs/version_{ID}/` at
`--output_base_dir`. Its contents are:

  - `hparams.yaml`: stores hyper-parameters of the model (used by `pytorch_lightning.LightningModule`)
  - `metrics.csv`: contains all metrics tracked during testing. Column IDs are identical to `metrics.csv` during
    [training](#training)
  - `test_masks`: directory containing segmentation masks obtained from U-Net

## Predicting
A trained U-Net can used for predictions with `cellseg_predict`. `cellseg_predict` returns only predicted masks
metrics and can be called with the following arguments:

  - `--help`: show help message
  - `--data`: Path to CSV file containing predict image file paths. The CSV file must have the columns `bf` and
    `--mask`. 
  - `--model`: Path to checkpoint file of trained pytorch_lightning.LightningModule.
  - `--suffix`: Suffix to append to all mask file names.
  - `--output_base_dir`: Path to output directory.
  - `--devices`: Devices to use for model training. If you want to use GPU(s) you have to provide `int` IDs. 
    Multiple GPU IDs have to be listed separated by spacebar (e.g. `2 5 9`). If multiple GPUs are provided only
    the first ID will be used. If you want to use the CPU you have to use "cpu". Default is "cpu".

Make sure to activate the virtual environment created during [installation](#installation) before calling
`cellseg_predict`.

Consider the following example:
```bash
# activate the virtual environment
conda activate cellseg

# generate CSV file for data
cellseg_generate_list \
  --bf "/path/to/predict_images/*.png" \
  --mask "/path/to/predict_images/*.png" # necessary to provide "--mask" for cellseg_generate_list \
  --out /path/to/output_directory \
  --prefix predict

# run prediction
cellseg_predict \
  --data /path/to/output_directory/predict_paths.csv \
  --model /path/to/checkpoint/file.ckpt \
  --suffix mask \
  --output_base_dir /path/to/results/folder \
  --devices 0 # predict with GPU 0
```

The output of `cellseg_predict` will be stored in subdirectories `lightning_logs/version_{ID}/` at
`--output_base_dir`. Its contents are:

  - `hparams.yaml`: stores hyper-parameters of the model (used by `pytorch_lightning.LightningModule`)
  - `predicted_masks`: directory containing segmentation masks obtained from U-Net

### napari plugin
`cellseg_predict` is also available as a plug-in for `napari` (link to napari-hub page and github page). 

## Image annotation tools
Available tools to annotate segmentations include:

  - [napari](https://napari.org/stable/)
  - [Labkit](https://imagej.net/plugins/labkit/) for [Fiji](https://imagej.net/software/fiji/downloads)
  - [QuPath](https://qupath.github.io)
  - [ilastik](https://www.ilastik.org)

## Troubleshooting & support
In case you are experiencing issues with `cellseg` inform us via the [issue tracker](https://github.com/CSDGroup/cell_segmentation/issues).
Before you submit an issue, check if it has been addressed in a previous issue.

## Citation
t.b.d.
