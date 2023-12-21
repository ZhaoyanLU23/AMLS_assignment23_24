# AMLS_assignment23_24

* [ ] TODO: Add repo description here.

## Files

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

### Solve tasks

* All options provided

```bash
$ python main.py --help
# TODO: add help
```

* Solve all tasks

```bash
$ python main.py solve --task all
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