#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

# hack here
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.logger import logger
from utils.dataset import Dataset

import numpy as np
import xgboost as xgb


class Solution:
    def __init__(self):
        pass

    def solve(self, data_abs_path: str):
        logger.info("=====================================")
        logger.info("|         Solving Task A ...        |")
        logger.info("=====================================")

        # TODO: add your solution here
        self.demo(data_abs_path)

        logger.info("----------Task A finished!-----------")

    def demo(self, path: str):
        """Cost 2s on CPU.
        [[144  90]
        [  7 383]]
        """
        from sklearn.metrics import confusion_matrix

        dataset = Dataset(path)
        logger.info("Read data finished!")

        clf = xgb.XGBClassifier()
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
