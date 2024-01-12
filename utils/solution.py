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
from sklearn.metrics import confusion_matrix, classification_report


class Solution:
    def __init__(
        self,
        dataset_path: str,
        device: str,
        config_path: str,
        save_result: bool = True,
        early_stopping_rounds: int = 3,
    ):
        self.dataset = Dataset(dataset_path)
        self.task_name = "Task N"
        self.device = device
        self.task_dir = ""
        self.config = {}
        self.save_result = save_result
        self.classifier = None
        self.target_names = None
        self.early_stopping_rounds = (
            None if early_stopping_rounds == 0 else early_stopping_rounds
        )
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

    @property
    def evals_result_path(self):
        evals_result_filename = f'{self.task_name.lower().replace(" ", "_")}_early_stopping_rounds_{self.early_stopping_rounds}_evals_result.json'
        path = os.path.join(self.task_dir, evals_result_filename)
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
        """Cross Validation"""
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
        """Training"""
        logger.info(f"[{self.task_name}] [Training] Running on {self.device}...")
        training_config: dict = self.config.get("training", {})
        random_state: int = self.config.get("random_state", DEFAULT_RANDOM_STATE)

        self.classifier = xgb.XGBClassifier(
            device=self.device,
            random_state=random_state,
            early_stopping_rounds=self.early_stopping_rounds,
            **training_config,
        )
        self.classifier.fit(
            self.dataset.old_X_train,
            self.dataset.old_y_train,
            eval_set=[
                (self.dataset.old_X_train, self.dataset.old_y_train),
                # the last entry in eval_set will be used for early stopping
                # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
                (self.dataset.X_val, self.dataset.y_val),
            ],
        )

        if self.save_result:
            import json

            # Save model into JSON format.
            self.classifier.save_model(self.training_model_path)
            logger.info(
                f"Model saved for {self.task_name} at {self.training_model_path}!"
            )
            # Save training set and val set results for plotting
            evals_result = self.classifier.evals_result()
            with open(self.evals_result_path, "w") as f:
                json.dump(evals_result, f)
            logger.info(
                f"Evals result saved for {self.task_name} at {self.evals_result_path}!"
            )

        train_score = self.classifier.score(self.dataset.X_train, self.dataset.y_train)
        logger.info(f"training score: {train_score}")

        y_pred = self.classifier.predict(self.dataset.X_train)
        mat = confusion_matrix(self.dataset.y_train, y_pred)
        logger.info(f"confusion matrix for training: \n{mat}")

        report = classification_report(
            self.dataset.y_train, y_pred, target_names=self.target_names
        )
        logger.info(f"classification report for training:\n{report}")

    def test(self):
        """Testing"""
        logger.info(f"[{self.task_name}] [Testing] Running on {self.device}...")

        if not self.classifier:
            logger.debug(f"Training stage skipped!")
            if os.path.exists(self.training_model_path):
                # load model from file
                self.classifier = xgb.XGBClassifier()
                self.classifier.load_model(self.training_model_path)
                logger.debug(
                    f"Model {self.classifier} loaded from {self.training_model_path}!"
                )
        assert (
            self.classifier is not None
        ), f"ERROR: No model exists! The model may be saved to {self.training_model_path}. Please use `--stages train` to train a model so that you can run testing."

        test_score = self.classifier.score(self.dataset.X_test, self.dataset.y_test)
        logger.info(f"testing score: {test_score}")

        y_pred = self.classifier.predict(self.dataset.X_test)
        mat = confusion_matrix(self.dataset.y_test, y_pred)
        logger.info(f"confusion matrix for testing: \n{mat}")

        report = classification_report(
            self.dataset.y_test, y_pred, target_names=self.target_names
        )
        logger.info(f"classification report for testing:\n{report}")
