[![DOI](https://zenodo.org/badge/496600864.svg)](https://zenodo.org/doi/10.5281/zenodo.10076235)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Tests](https://github.com/CSDGroup/aisegcell/workflows/tests/badge.svg)](https://github.com/CSDGroup/aisegcell/actions)
[![codecov](https://codecov.io/gh/CSDGroup/aisegcell/branch/main/graph/badge.svg?token=63T8R6MUMB)](https://codecov.io/gh/CSDGroup/aisegcell)

# aiSEGcell - Overview
This repository contains a `torch` implementation of U-Net ([Ronneberger et al., 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)).
We provide [trained](#trained-models) models to semantically segment nuclei and whole cells in bright field images.
Please cite [this paper](#citation) if you are using this code in your research.

## Contents
  - [Installation](#installation)
    - [Virtual environment](#virtual-environment-setup)
    - [pip installation](#pip-installation)
    - [Source installation](#source-installation)
  - [Data](#data)
  - [Training](#training)
    - [Trained models](#trained-models)
  - [Testing](#testing)
  - [Predicting](#predicting)
    - [napari plugin](#napari-plugin)
  - [Image annotation tools](#image-annotation-tools)
  - [Troubleshooting & support](#troubleshooting-&-support)
  - [Citation](#citation)

## Installation
If you do not have python installed already, we recommend installing it using the
[Anaconda distribution](https://www.anaconda.com/products/distribution). `aisegcell` was tested with `python 3.8.6`.

### Virtual environment setup
If you do not use and IDE that handles [virtual environments](https://realpython.com/python-virtual-environments-a-primer/)
for you (e.g. [PyCharm](https://www.jetbrains.com/pycharm/)) use your command line application (e.g. `Terminal`) and
one of the many virtual environment tools (see [here](https://testdriven.io/blog/python-environments/)). We will
use `conda`

1) Create new virtual environment

    ```bash
    conda create -n aisegcell python=3.8.6
    ```

2) Activate virtual environment

    ```bash
    conda activate aisegcell
    ```

### pip installation
Recommended if you do not want to develop the `aisegcell` code base.

3) Install `aisegcell`
    ```bash
    # update pip
    pip install -U pip==23.2.1
    pip install aisegcell
    ```

4) (Optional) `GPUs` greatly speed up training and inference of U-Net and are available for `torch` (`v1.10.2`) for
`Windows` and `Linux`. Check if your `GPU(s)` are CUDA compatible
([`Windows`](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#verify-you-have-a-cuda-capable-gpu),
 [`Linux`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#verify-you-have-a-cuda-capable-gpu)) and
 update their drivers if necessary.

5) [Install `torch`/`torchvision`](https://pytorch.org/get-started/previous-versions/) compatible with your system.
`aisegcell` was tested with `torch` version `1.10.2`, `torchvision` version `0.11.3`, and `cuda` version
`11.3.1`. Depending on your OS, your `CPU` or `GPU` (and `CUDA` version) the installation may change

```bash
# Windows/Linux CPU
pip install torch==1.10.2+cpu torchvision==0.11.3+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Windows/Linux GPU (CUDA 11.3.X)
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# macOS CPU
pip install torch==1.10.2 torchvision==0.11.3

```

6) [Install `pytorch-lightning`](https://www.pytorchlightning.ai). `aisegcell` was tested with version `1.5.9`.

```bash
# note the installation of v1.5.9 does not use pip install lightning
pip install pytorch-lightning==1.5.9
```


### Source installation
Installation requires a command line application (e.g. `Terminal`) with
[`git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [python](https://www.python.org) installed.
If you operate on `Windows` we recommend using
[`Ubuntu on Windows`](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview).
Alternatively, you can install [`Anaconda`](https://docs.anaconda.com/anaconda/user-guide/getting-started/) and
use `Anaconda Powershell Prompt`. An introductory tutorial on how to use `git` and GitHub can be found
[here](https://www.w3schools.com/git/default.asp?remote=github).

3) (Optional) If you use `Anaconda Powershell Prompt`, install `git` through `conda`

    ```bash
    conda install -c anaconda git
    ```

4) clone the repository (consider `ssh` alternative)

    ```bash
    # change directory
    cd /path/to/directory/to/clone/repository/to

    git clone https://github.com/CSDGroup/aisegcell.git
    ```

5) Navigate to the cloned directory

    ```bash
    cd aisegcell
    ```

6) Install `aisegcell`
    ```bash
    # update pip
    pip install -U pip==23.2.1
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
`Windows` and `Linux`. Check if your `GPU(s)` are CUDA compatible
([`Windows`](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#verify-you-have-a-cuda-capable-gpu),
 [`Linux`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#verify-you-have-a-cuda-capable-gpu)) and
 update their drivers if necessary.

8) [Install `torch`/`torchvision`](https://pytorch.org/get-started/previous-versions/) compatible with your system.
`aisegcell` was tested with `torch` version `1.10.2`, `torchvision` version `0.11.3`, and `cuda` version
`11.3.1`. Depending on your OS, your `CPU` or `GPU` (and `CUDA` version) the installation may change

```bash
# Windows/Linux CPU
pip install torch==1.10.2+cpu torchvision==0.11.3+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Windows/Linux GPU (CUDA 11.3.X)
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# macOS CPU
pip install torch==1.10.2 torchvision==0.11.3

```

9) [Install `pytorch-lightning`](https://www.pytorchlightning.ai). `aisegcell` was tested with version `1.5.9`.

```bash
# note the installation of v1.5.9 does not use pip install lightning
pip install pytorch-lightning==1.5.9
```

## Data
U-Net is currently intended for single-class semantic segmentation. Input images are expected to be 8-bit or
16-bit greyscale images. Segmentation masks are expected to decode background as 0 intensity and all intensities
\>0 are converted to a single intensity value (255). Consequently, different instances of a class (instance
segmentation) or multi-class segmentations are handled as single-class segmentations. Have a look at
[this notebook](https://github.com/CSDGroup/aisegcell/blob/main/notebooks/data_example.ipynb)
for a data example.


## Training
Training U-Net is as simple as calling the command `aisegcell_train`. We provide a
[notebook](https://github.com/CSDGroup/aisegcell/blob/main/notebooks/unet_example.ipynb) on how to train
U-Net with a minimal working example. `aisegcell_train` is available if you activate the virtual environment you
[installed](#installation) and can be called with the following arguments:

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

The command `aisegcell_generate_list` can be used to write CSV files for `--data` and `--data_val` and
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
conda activate aisegcell

# generate CSV files for data and data_val
aisegcell_generate_list \
  --bf "/path/to/train_images/*/*.png" # i.e. select all PNG files in all sub-directories of /path/to/train_images\
  --mask "/path/to/train_masks/*/*mask.png" # i.e. select all files in all sub-directories that end with "mask.png"\
  --out /path/to/output_directory \
  --prefix train

aisegcell_generate_list \
  --bf "/path/to/val_images/*.png" \
  --mask "/path/to/val_masks/*.png" \
  --out /path/to/output_directory \
  --prefix val

# starting multi-GPU training
aisegcell_train \
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
aisegcell_train \
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

The output of `aisegcell_train` will be stored in subdirectories `{DATE}_Unet_{ID1}/lightning_logs/version_{ID2}/` at
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
    `--checkpoint` of `aisegcell_train` or `--model` of `aisegcell_test` and `aisegcell_predict`.
    - `best-f1-epoch={EPOCH}-step={STEP}.ckpt`: model weights with the (currently) highest `f1_epoch`
    - `best-iou-epoch={EPOCH}-step={STEP}.ckpt`: model weights with the (currently) highest `iou_epoch`
    - `best-loss-epoch={EPOCH}-step={STEP}.ckpt`: model weights with the (currently) lowest `loss_val_epoch`
    - `latest-epoch={EPOCH}-step={STEP}.ckpt`: model weights of the (currently) latest checkpoint

### Trained models
We provide trained models:

| modality | image format | example image | description | availability |
| :-- | :-: | :-: | :-: | :-- |
| nucleus segmentation | 2D grayscale | <img src="https://github.com/CSDGroup/aisegcell/raw/main/images/nucseg.png" title="example nucleus segmentation" width="180px" align="center"> | Trained on a data set (link to data set) of 9849 images (~620k nuclei). | [ETH Research Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/608641) |
| whole cell segmentation | 2D grayscale | <img src="https://github.com/CSDGroup/aisegcell/raw/main/images/cellseg.png" title="example whole cell segmentation" width="180px" align="center"> | Trained on a data set (link to data set) of 224 images (~12k cells). | [ETH Research Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/608646) |

## Testing
A trained U-Net can be tested with `aisegcell_test`. We provide a
[notebook](https://github.com/CSDGroup/aisegcell/blob/main/notebooks/unet_example.ipynb) on how to test
with U-Net. `aisegcell_test` returns predicted masks and performance metrics. `aisegcell_test` can be called with the
following arguments:

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
`aisegcell_test`.

Consider the following example:
```bash
# activate the virtual environment
conda activate aisegcell

# generate CSV file for data
aisegcell_generate_list \
  --bf "/path/to/test_images/*.png" \
  --mask "/path/to/test_masks/*.png" \
  --out /path/to/output_directory \
  --prefix test

# run testing
aisegcell_test \
  --data /path/to/output_directory/test_paths.csv \
  --model /path/to/checkpoint/file.ckpt \
  --suffix mask \
  --output_base_dir /path/to/results/folder \
  --devices 0 # predict with GPU 0
```

The output of `aisegcell_test` will be stored in subdirectories `lightning_logs/version_{ID}/` at
`--output_base_dir`. Its contents are:

  - `hparams.yaml`: stores hyper-parameters of the model (used by `pytorch_lightning.LightningModule`)
  - `metrics.csv`: contains all metrics tracked during testing. Column IDs are identical to `metrics.csv` during
    [training](#training)
  - `test_masks`: directory containing segmentation masks obtained from U-Net

## Predicting
A trained U-Net can used for predictions with `aisegcell_predict`. We provide a
[notebook](https://github.com/CSDGroup/aisegcell/blob/main/notebooks/unet_example.ipynb) on how to
predict with U-Net. `aisegcell_predict` returns only predicted masks metrics and can be called with the following
arguments:

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
`aisegcell_predict`.

Consider the following example:
```bash
# activate the virtual environment
conda activate aisegcell

# generate CSV file for data
aisegcell_generate_list \
  --bf "/path/to/predict_images/*.png" \
  --mask "/path/to/predict_images/*.png" # necessary to provide "--mask" for aisegcell_generate_list \
  --out /path/to/output_directory \
  --prefix predict

# run prediction
aisegcell_predict \
  --data /path/to/output_directory/predict_paths.csv \
  --model /path/to/checkpoint/file.ckpt \
  --suffix mask \
  --output_base_dir /path/to/results/folder \
  --devices 0 # predict with GPU 0
```

The output of `aisegcell_predict` will be stored in subdirectories `lightning_logs/version_{ID}/` at
`--output_base_dir`. Its contents are:

  - `hparams.yaml`: stores hyper-parameters of the model (used by `pytorch_lightning.LightningModule`)
  - `predicted_masks`: directory containing segmentation masks obtained from U-Net

### napari plugin
`aisegcell_predict` is also available as a plug-in for `napari` (link to napari-hub page and github page).

## Image annotation tools
Available tools to annotate segmentations include:

  - [napari](https://napari.org/stable/)
  - [Labkit](https://imagej.net/plugins/labkit/) for [Fiji](https://imagej.net/software/fiji/downloads)
  - [QuPath](https://qupath.github.io)
  - [ilastik](https://www.ilastik.org)

## Troubleshooting & support
In case you are experiencing issues with `aisegcell` inform us via the [issue tracker](https://github.com/CSDGroup/aisegcell/issues).
Before you submit an issue, check if it has been addressed in a previous issue.

## Citation
t.b.d.
