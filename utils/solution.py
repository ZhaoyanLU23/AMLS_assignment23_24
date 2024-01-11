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

from constants import N_KFOLD, DEFAULT_RANDOM_STATE

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV


class Solution:
    def __init__(
        self, dataset_path: str, device: str, config_path: str, save_result: bool = True
    ):
        self.dataset = Dataset(dataset_path)
        self.task_name = "Task N"
        self.device = device
        self.task_dir = ""
        self.config = {}
        self.save_result = save_result
        self.classifier = None
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)

    @property
    def training_model_path(self):
        training_model_filename = (
            f'{self.task_name.lower().replace(" ", "_")}_training_model.json'
        )
        path = os.path.join(self.task_dir, training_model_filename)
        return path

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
        val_config: dict = self.config.get("validation", {})
        param_grid: dict = val_config.get("param_grid", {})
        random_state: int = self.config.get("random_state", DEFAULT_RANDOM_STATE)
        logger.info(f"Searching xgboost params: {param_grid}...")

        # We set n_jobs to None here for cross validation using scikit-learn
        # See: https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html#number-of-parallel-threads
        # And https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html#reducing-memory-usage
        base_estimator = xgb.XGBClassifier(
            device=self.device, n_jobs=None, random_state=random_state
        )
        # We use StratifiedKFold here for the imbalance in the distribution of the target classes
        skf = StratifiedKFold(n_splits=N_KFOLD)

        X_train = self.dataset.X_train
        y_train = self.dataset.y_train

        sh = GridSearchCV(
            base_estimator,
            param_grid,
            cv=skf,
            n_jobs=1,
        ).fit(X_train, y_train)
        logger.info(sh.best_estimator_)
        logger.info(sh.best_score_)
        logger.info(sh.best_params_)

        if self.save_result:
            from pandas import DataFrame

            df = DataFrame(sh.cv_results_)
            # Get current date and time
            datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
            cv_result_filename = f'{datetime_str}_{self.task_name.lower().replace(" ", "_")}_cross_validation.csv'
            cv_result_path = os.path.join(self.task_dir, cv_result_filename)
            df.to_csv(cv_result_path)

    def train(self):
        logger.info(f"[{self.task_name}] [Training] Running on {self.device}...")
        training_config: dict = self.config.get("training", {})
        random_state: int = self.config.get("random_state", DEFAULT_RANDOM_STATE)
        self.classifier = xgb.XGBClassifier(
            device=self.device, random_state=random_state, **training_config
        )
        self.classifier.fit(self.dataset.X_train, self.dataset.y_train)
        if self.save_result:
            # Save model into JSON format.
            self.classifier.save_model(self.training_model_path)
        train_score = self.classifier.score(self.dataset.X_train, self.dataset.y_train)
        logger.info(f"training score: {train_score}")

    def test(self):
        logger.info(f"[{self.task_name}] [Testing] Running on {self.device}...")
        if not self.classifier:
            if os.path.exists(self.training_model_path):
                # load model from file
                with open(self.training_model_path, "r") as f:
                    model_json = json.load(f)
                self.classifier = xgb.XGBClassifier(model_json)
        assert (
            self.classifier
        ), "No model exists! Please use `--stages train` to train a model so that you can run testing."
        test_score = self.classifier.score(self.dataset.X_test, self.dataset.y_test)
        logger.info(f"testing score: {test_score}")
