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
$ tree
.
├── A
│   └── solution.py             # Solution for task A
├── B
│   └── solution.py             # Solution for task B
├── Datasets                    # An empty dir for Datasets
├── Makefile
├── README.md
├── environment.yml             # conda environment config
├── main.py                     # Entrypoint of this repo
└── requirements.txt            # python package requirements
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
$ cat requirements.txt
# format python files
black
# read and process datasets
numpy
```

## Usage

### Setup

* Install all packages needed using `conda`:

```bash
$ make create-env
...

$ conda activate amls-final-zhaoyanlu
```

* (optional) You can use `pip` instead:

```bash
$ pip install -f requirements.txt
```

* All actions provided

```bash
$ python main.py --help
usage: main.py [-h] [-v] {solve,info} ...

AMLS Final Assignment

positional arguments:
  {solve,info}  actions provided

options:
  -h, --help    show this help message and exit
  -v            verbose

$ python main.py solve --help
usage: main.py solve [-h] [--task TASK]

options:
  -h, --help   show this help message and exit
  --task TASK  task to solve: A, B, or all; default: all
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
$ python main.py solve
# TODO: only training
```

* Task A

```bash
# Solve Task A
$ python main.py solve --task A
# TODO: only training
```

* Task B

```bash
# Solve Task B
$ python main.py solve --task B
# TODO: only training
```