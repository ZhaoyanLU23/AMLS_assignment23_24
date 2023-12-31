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
    def __init__(self, dataset_path: str, device: str, config_path: str):
        super().__init__(
            dataset_path=dataset_path, device=device, config_path=config_path
        )
        self.task_name = "Task B"
        self.task_dir = TASK_B_DIR

    def val(self):
        super().val()

    def train(self):
        super().train()

    def test(self):
        super().test()


# 12/31 01:27:44 [INFO]: Searching xgboost params: {'learning_rate': [0.2], 'max_depth': [5], 'min_child_weight': [1], 'gamma': [0], 'subsample': [0.6], 'colsample_bytree': [0.6]}...
# /home/uceezl8/.conda/envs/amls-final-zhaoyanlu/lib/python3.9/site-packages/xgboost/core.py:160: UserWarning: [01:27:59] WARNING: /home/conda/feedstock_root/build_artifacts/xgboost-split_1703076482591/work/src/common/error_msg.cc:58: Falling back to prediction using DMatrix due to mismatched devices. This might lead to higher memory usage and slower performance. XGBoost is running on: cuda:1, while the input data is on: cpu.
# Potential solutions:
# - Use a data structure that matches the device ordinal in the booster.
# - Set the device for booster before call to inplace_predict.

# This warning will only be shown once.

#   warnings.warn(smsg, UserWarning)
# 12/31 01:29:16 [INFO]: XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=0.6, device='cuda:1', early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=0, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.2, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=5, max_leaves=None,
#               min_child_weight=1, missing=nan, monotone_constraints=None,
#               multi_strategy=None, n_estimators=None, n_jobs=None,
#               num_parallel_tree=None, objective='multi:softprob', ...)
# 12/31 01:29:16 [INFO]: 0.75185
# 12/31 01:29:16 [INFO]: {'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.2, 'max_depth': 5, 'min_child_weight': 1, 'subsample': 0.6}
