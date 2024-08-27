# dynSGL
Dynamic Signed Graph Learning from Smooth Temporal Data

## Installation
To install the package, first create a virtual environment with a Python version at least `3.12`. With `conda`, this can be done by: 
```bash
$ conda create -n mvgl_env -c conda-forge python=3.12
```
Then, use `pip` to install the package from Github. 
```bash
$ pip install git+https://github.com/SPLab-aviyente/dynSGL.git
```

## Usage

See files under `scripts/` for how to run the proposed method or check docstring of `learn_a_dynamic_signed_graph()` as follows:
```bash
>>> from dynsgl import graphlearning
>>> help(graphlearning.learn_a_dynamic_signed_graph)