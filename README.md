# AMLS_assignment23_24

* Name: Zhaoyan LU
* Student ID: 23049710

## Contents

- [Overview](#Overview)
- [Repo Structure](#Repo-Structure)
- [Datasets](#Datasets)
- [Requirements](#Requirements)
- [Usage](#Usage)
    - [Setup](#Setup)
    - [Solve Tasks](#Solve-Tasks)

## Overview

* [ ] TODO: Add repo description here.

## Repo Structure

* [ ] TODO: Explain the role of each file.

```bash
# TODO: update repo structure
$ tree
.
├── A
│   ├── README.md
│   ├── config.json
│   ├── solution.py                 # Solution for task A
│   └── task_a_training_model.json  # Model saved
├── B
│   ├── README.md
│   ├── config.json
│   ├── solution.py                 # Solution for task B
│   └── task_b_training_model.json  # Model saved
├── Datasets                        # An empty dir for Datasets
├── Makefile
├── README.md
├── constants.py                    # global constants
├── environment.yml                 # conda environment config
├── main.py                         # Entrypoint of this repo
├── requirements.txt                # python package requirements
└── utils
    ├── dataset.py                  # Dataset Loader
    ├── logger.py                   # Logger for this project
    └── solution.py                 # Solution framework for each task
```

## Datasets

Datasets being used are downloaded from [zenodo](https://zenodo.org/records/6496656):

* [PneumoniaMNIST](https://zenodo.org/records/6496656/files/pneumoniamnist.npz)
* [PathMNIST](https://zenodo.org/records/6496656/files/pathmnist.npz)

Both of them are `.npz` files.

You can use the following command to download them:

```bash
$ make download
...

$ tree Datasets
Datasets
├── PathMNIST
│   └── pathmnist.npz
└── PneumoniaMNIST
    └── pneumoniamnist.npz

2 directories, 2 files
```

## Requirements

* [ ] TODO: Add assignment requirements.

```bash
# TODO: update requirements.txt
$ cat requirements.txt
# format python files
black
# read and process datasets
numpy
# Models
scikit-learn
# GPU version
# If you want to use CPU instead, use py-xgboost-cpu
py-xgboost-gpu
# save results
pandas
```

**NOTE**: `py-xgboost-gpu` will use GPU. If you want to use CPU only, use [`py-xgboost-cpu`](https://xgboost.readthedocs.io/en/stable/install.html#conda) and set the device using `--device cpu`.

## Usage

### Setup

* Install all packages needed using `conda`:

```bash
$ make create-env
...

$ conda activate amls-final-zhaoyanlu
```

* Or you can use `pip` instead:

```bash
$ pip install -f requirements.txt
```

* All actions provided

```bash
$ python main.py --help
usage: main.py [-h] [-d] [-v] {solve,info} ...

AMLS Final Assignment

positional arguments:
  {solve,info}   actions provided

optional arguments:
  -h, --help     show this help message and exit
  -d, --debug    Print lots of debugging statements
  -v, --verbose  Be verbose

# options for the solve action
$ python main.py solve --help
usage: main.py solve [-h] [--task TASK] [--stages STAGES]
                     [--device DEVICE] [--save SAVE]
                     [--early_stopping_rounds EARLY_STOPPING_ROUNDS]

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           task to solve: A, B, or all; default: all
  --stages STAGES       task stages: val, train, test, or all;
                        default: val,train,test
  --device DEVICE       Device ordinal for xgboost, available options:
                        cpu, cuda, and gpu; default: cuda
  --save SAVE           Save results or not, available options: True
                        or False; default: True
  --early_stopping_rounds EARLY_STOPPING_ROUNDS
                        Used to prevent overfitting; require >= 0,
                        default: 3
```

* Show info

```bash
$ python main.py info
-------------------------------------
|       AMLS Assignment 23-24       |
|         Name: Zhaoyan Lu          |
|        Student No: 23049710       |
-------------------------------------
```

### Solve Tasks

* Solve all tasks

```bash
# Run all stages for all tasks
$ python main.py -v solve
# only training
$ python main.py -v solve --stages train
# training and testing
$ python main.py -v solve --stages train,test
```

* Task A

```bash
# Solve Task A
$ python main.py -v solve --task A
# only training
$ python main.py -v solve --task A --stages train
# training and testing
$ python main.py -v solve --task A --stages train,test
```

* Task B

```bash
# Solve Task B
$ python main.py -v solve --task B
# only training
$ python main.py -v solve --task A --stages train
# training and testing
$ python main.py -v solve --task B --stages train,test
```
