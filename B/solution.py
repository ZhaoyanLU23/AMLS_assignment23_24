#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from typing import List

# hack here
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.dataset import Dataset
from utils.logger import logger

import numpy as np
import xgboost as xgb


class Solution:
    def __init__(self, path: str):
        self.dataset = Dataset(path)

    def solve(self, stages: List[str]):
        logger.info("=====================================")
        logger.info("|         Solving Task B ...        |")
        logger.info("=====================================")

        if "val" in stages:
            self.val()
        if "train" in stages:
            self.train()
        if "test" in stages:
            self.test()

        logger.info("----------Task B finished!-----------")

    def val(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def demo(self, path: str):
        """Cost one hour on CPU.
        [[1004    1    5    0   32  284    7    0    5]
        [   0  847    0    0    0    0    0    0    0]
        [   0    0  121    7    0  152    1   52    6]
        [   0    1   10  365  118    2   79    1   58]
        [  36   83    1   15  852    1   25    2   20]
        [   0    0  161    1    0  281   49   84   16]
        [   4    0    7   16   25   12  475    0  202]
        [   0    0  109    0    6   65   14  162   65]
        [   0    0   28   30   14   10  154    4  993]]
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
