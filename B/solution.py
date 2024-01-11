#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from typing import List

# hack here
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.solution import Solution
from constants import TASK_B_DIR

import numpy as np
import xgboost as xgb


class SolutionB(Solution):
    def __init__(
        self, dataset_path: str, device: str, config_path: str, save_result: bool = True
    ):
        super().__init__(
            dataset_path=dataset_path,
            device=device,
            config_path=config_path,
            save_result=save_result,
        )
        self.task_name = "Task B"
        self.task_dir = TASK_B_DIR

    def val(self):
        # 12/31 02:06:55 [INFO]: Searching xgboost params: {'learning_rate': [0.1, 0.2, 0.5], 'max_depth': [5, 7, 9], 'min_child_weight': [1, 3, 5, 7], 'gamma': [0], 'subsample': [0.3, 0.5, 0.7], 'colsample_bytree': [0.3, 0.5, 0.7]}...
        # 12/31 12:34:27 [INFO]: XGBClassifier(base_score=None, booster=None, callbacks=None,
        #               colsample_bylevel=None, colsample_bynode=None,
        #               colsample_bytree=0.7, device='cuda:1', early_stopping_rounds=None,
        #               enable_categorical=False, eval_metric=None, feature_types=None,
        #               gamma=0, grow_policy=None, importance_type=None,
        #               interaction_constraints=None, learning_rate=0.2, max_bin=None,
        #               max_cat_threshold=None, max_cat_to_onehot=None,
        #               max_delta_step=None, max_depth=9, max_leaves=None,
        #               min_child_weight=7, missing=nan, monotone_constraints=None,
        #               multi_strategy=None, n_estimators=None, n_jobs=None,
        #               num_parallel_tree=None, objective='multi:softprob', ...)
        # 12/31 12:34:27 [INFO]: 0.77423
        # 12/31 12:34:27 [INFO]: {'colsample_bytree': 0.7, 'gamma': 0, 'learning_rate': 0.2, 'max_depth': 9, 'min_child_weight': 7, 'subsample': 0.7}
        super().val()

    def train(self):
        super().train()

    def test(self):
        super().test()
