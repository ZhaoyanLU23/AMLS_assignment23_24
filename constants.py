#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

CWD = os.getcwd()

TASK_A_DIR = "A"
TASK_B_DIR = "B"

# Dataset path for each task
DATASET_DIR = "Datasets"
TASK_A_DATA_ABS_PATH = os.path.join(
    CWD,
    DATASET_DIR,
    "PneumoniaMNIST",
    "pneumoniamnist.npz",
)
TASK_B_DATA_ABS_PATH = os.path.join(
    CWD,
    DATASET_DIR,
    "PathMNIST",
    "pathmnist.npz",
)

# config for each task
CONFIG_FILENAME = "config.json"
TASK_A_CONFIG_PATH = os.path.join(
    CWD,
    TASK_A_DIR,
    CONFIG_FILENAME,
)
TASK_B_CONFIG_PATH = os.path.join(
    CWD,
    TASK_B_DIR,
    CONFIG_FILENAME,
)

# Random state
DEFAULT_RANDOM_STATE = 20232024

# N splits for KFold
N_KFOLD = 10
