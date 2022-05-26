# cell_segmentation
This repository contains a pipeline to segment cell bodies.

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

6) [Install `torch`](https://pytorch.org/get-started/locally/) compatible with your system. cell_segmentation was
tested with `torch` version `1.10.2` and `cuda` version `11.3.1`.

7) [Install `pytorch-lightning`](https://www.pytorchlightning.ai) compatible with your system. cell_segmentation
was tested with version `1.5.9`.

