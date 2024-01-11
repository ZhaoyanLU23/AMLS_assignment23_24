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
    def __init__(
        self, dataset_path: str, device: str, config_path: str, save_result: bool = True
    ):
        super().__init__(
            dataset_path=dataset_path,
            device=device,
            config_path=config_path,
            save_result=save_result,
        )
        self.task_name = "Task A"
        self.task_dir = TASK_A_DIR
        # target_names come from: https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py
        self.target_names = ["normal", "pneumonia"]

    def val(self):
        # 12/31 01:26:58 [INFO]: XGBClassifier(base_score=None, booster=None, callbacks=None,
        #               colsample_bylevel=None, colsample_bynode=None,
        #               colsample_bytree=0.6, device='cuda:1', early_stopping_rounds=None,
        #               enable_categorical=False, eval_metric=None, feature_types=None,
        #               gamma=0, grow_policy=None, importance_type=None,
        #               interaction_constraints=None, learning_rate=0.2, max_bin=None,
        #               max_cat_threshold=None, max_cat_to_onehot=None,
        #               max_delta_step=None, max_depth=5, max_leaves=None,
        #               min_child_weight=1, missing=nan, monotone_constraints=None,
        #               multi_strategy=None, n_estimators=None, n_jobs=None,
        #               num_parallel_tree=None, random_state=20232024, ...)
        # 12/31 01:26:58 [INFO]: 0.9640666860245334
        # 12/31 01:26:58 [INFO]: {'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.2, 'max_depth': 5, 'min_child_weight': 1, 'subsample': 0.6}
        super().val()

    def train(self):
        # 01/11 04:30:03 [INFO]: training score: 1.0
        # 01/11 04:30:03 [INFO]: confusion matrix for training:
        # [[1349    0]
        #  [   0 3883]]
        # 01/11 04:30:03 [INFO]: classification report for training:
        #               precision    recall  f1-score   support

        #       normal       1.00      1.00      1.00      1349
        #    pneumonia       1.00      1.00      1.00      3883

        #     accuracy                           1.00      5232
        #    macro avg       1.00      1.00      1.00      5232
        # weighted avg       1.00      1.00      1.00      5232
        super().train()

    def test(self):
        # 01/11 04:30:03 [INFO]: testing score: 0.842948717948718
        # 01/11 04:30:03 [INFO]: confusion matrix for testing:
        # [[141  93]
        #  [  5 385]]
        # 01/11 04:30:03 [INFO]: classification report for testing:
        #               precision    recall  f1-score   support

        #       normal       0.97      0.60      0.74       234
        #    pneumonia       0.81      0.99      0.89       390

        #     accuracy                           0.84       624
        #    macro avg       0.89      0.79      0.81       624
        # weighted avg       0.87      0.84      0.83       624
        super().test()
