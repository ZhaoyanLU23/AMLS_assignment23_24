#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from typing import List

# hack here
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import logger
from utils.solution import Solution
from constants import TASK_A_DIR

import numpy as np
import xgboost as xgb


class SolutionA(Solution):
    def __init__(self, dataset_path: str, device: str, config_path: str):
        super().__init__(
            dataset_path=dataset_path, device=device, config_path=config_path
        )
        self.task_name = "Task A"
        self.task_dir = TASK_A_DIR

    def val(self):
        super().val()

    def train(self):
        super().train()

    def test(self):
        super().test()

    # def demo(self, path: str):
    #     """Cost 2s on CPU.
    #     [[144  90]
    #     [  7 383]]
    #     """
    #     from sklearn.metrics import confusion_matrix

    #     dataset = Dataset(path)
    #     logger.info("Read data finished!")

    #     clf = xgb.XGBClassifier(verbosity=2, device="cuda", n_jobs=3)
    #     logger.info("Start fitting...")
    #     # Fit the model, test sets are used for early stopping.
    #     clf.fit(dataset.X_train, dataset.y_train)
    #     logger.info("After fitting...")
    #     # Save model into JSON format.
    #     # clf.save_model("clf.json")
    #     predictions = clf.predict(dataset.X_test)
    #     actuals = dataset.y_test
    #     logger.info(confusion_matrix(actuals, predictions))
    #     logger.info("Model saved!")
