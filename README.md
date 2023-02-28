[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Test](https://github.com/stardist/stardist/workflows/Test/badge.svg)](https://github.com/stardist/stardist/actions?query=workflow%3ATest)

# *CellSeg* - Overview
This repository contains a `torch` implementation of U-Net ([Ronneberger et al., 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)). Please cite [this paper](#citation) if you are using
this code in your research.

## Contents
  - [Installation](#installation)
  - [Training](#training)
    - [Pretrained models](#pretrained-models)
  - [Testing](#testing)
  - [Predicting](#predicting)
    - [napari plugin](#napari-plugin)
  - [Other resources](#other-resources)
    - [image annotation tools](#image-annotation-tools)
  - [Troubleshooting & support](#troubleshooting-&-support)
  - [Citation](#citation)

## Installation
1) clone the repository (consider `ssh` alternative)

    ```bash
    git clone https://github.com/dsethz/cell_segmentation.git
    ```

2) Navigate to the cloned directory

    ```bash
    cd cell_segmentation
    ```

3) We recommend using a virtual environment. If you are using anaconda, you can use the following.

    ```bash
    conda create -n cellseg python=3.8
    ```

4) Activate your virtual environment, after it has been created.

    ```bash
    conda activate cellseg
    ```

5) Install `cell_segmentation`.

    1) as a user

        ```bash
        pip install .
        ```
    2) as a developer (in editable mode with development dependencies and pre-commit hooks)
 
        ```bash
        pip install -e ".[dev]"
        pre-commit install
        ```

6) [Install `torch`/`torchvision`](https://pytorch.org/get-started/locally/) compatible with your system. cell_segmentation was
tested with `torch` version `1.10.2`, `torchvision` version `0.11.3`, and `cuda` version `11.3.1`.

7) [Install `pytorch-lightning`](https://www.pytorchlightning.ai) compatible with your system. cell_segmentation
was tested with version `1.5.9`.

## Training

### Pretrained models

## Testing

## Predicting

### napari plugin

## Other resources

### Image annotation tools

## Troubleshooting & support


## Citation
t.b.d.
