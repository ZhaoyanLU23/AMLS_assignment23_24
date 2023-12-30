#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import sys
from datetime import datetime
from typing import List

from .dataset import Dataset
from .logger import logger

# hack here
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from constants import N_KFOLD, TASK_A_DIR

import xgboost as xgb
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold, GridSearchCV


class Solution:
    def __init__(self, dataset_path: str, device: str, config_path: str):
        self.dataset = Dataset(dataset_path)
        self.task_name = "Task N"
        self.device = device
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)

    def solve(self, stages: List[str]):
        logger.info("=====================================")
        logger.info(f"          Solving {self.task_name} ...         ")
        logger.info("=====================================")

        if "val" in stages:
            self.val()
        if "train" in stages:
            self.train()
        if "test" in stages:
            self.test()

        logger.info(f"----------{self.task_name} finished!-----------")

    def val(self):
        logger.info(f"[{self.task_name}] [Validation] Running on {self.device}...")
        val_config = self.config.get("validation", {})
        param_grid = val_config.get("param_grid", {})
        logger.info(f"Searching xgboost params: {param_grid}...")
        # We set n_jobs to None here for cross validation using scikit-learn
        # See: https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html#number-of-parallel-threads
        base_estimator = xgb.XGBClassifier(device=self.device, n_jobs=None)
        # We use StratifiedKFold here for the imbalance in the distribution of the target classes
        skf = StratifiedKFold(n_splits=N_KFOLD)
        sh = GridSearchCV(
            base_estimator,
            param_grid,
            cv=skf,
            n_jobs=1,
        ).fit(self.dataset.X_train, self.dataset.y_train)
        logger.info(sh.best_estimator_)
        logger.info(sh.best_score_)
        logger.info(sh.best_params_)
        df = DataFrame(sh.cv_results_)
        # Get current date and time
        datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
        cv_result_filename = f'{datetime_str}_{self.task_name.lower().replace(" ", "_")}_cross_validation.csv'
        cv_result_path = os.path.join(TASK_A_DIR, cv_result_filename)
        df.to_csv(cv_result_path)

    def train(self):
        logger.info(f"[{self.task_name}] [Training] Running on {self.device}...")

    def test(self):
        logger.info(f"[{self.task_name}] [Testing] Running on {self.device}...")
