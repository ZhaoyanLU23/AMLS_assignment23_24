#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from typing import List

# hack here
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import logger
from utils.dataset import Dataset

import numpy as np
import xgboost as xgb


class Solution:
    def __init__(self, path: str):
        self.dataset = Dataset(path)

    def solve(self, stages: List[str]):
        logger.info("=====================================")
        logger.info("|         Solving Task A ...        |")
        logger.info("=====================================")

        if "val" in stages:
            self.val()
        if "train" in stages:
            self.train()
        if "test" in stages:
            self.test()

        logger.info("----------Task A finished!-----------")

    def val(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def demo(self, path: str):
        """Cost 2s on CPU.
        [[144  90]
        [  7 383]]
        """
        from sklearn.metrics import confusion_matrix

        dataset = Dataset(path)
        logger.info("Read data finished!")

        clf = xgb.XGBClassifier(verbosity=2, device="cuda", n_jobs=3)
        logger.info("Start fitting...")
        # Fit the model, test sets are used for early stopping.
        clf.fit(dataset.X_train, dataset.y_train)
        logger.info("After fitting...")
        # Save model into JSON format.
        # clf.save_model("clf.json")
        predictions = clf.predict(dataset.X_test)
        actuals = dataset.y_test
        logger.info(confusion_matrix(actuals, predictions))
        logger.info("Model saved!")
