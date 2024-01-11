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
        self,
        dataset_path: str,
        device: str,
        config_path: str,
        save_result: bool = True,
        early_stopping_rounds: int = True,
    ):
        super().__init__(
            dataset_path=dataset_path,
            device=device,
            config_path=config_path,
            save_result=save_result,
            early_stopping_rounds=early_stopping_rounds,
        )
        self.task_name = "Task B"
        self.task_dir = TASK_B_DIR
        # target_names come from: https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py
        self.target_names = (
            "adipose",
            "background",
            "debris",
            "lymphocytes",
            "mucus",
            "smooth muscle",
            "normal colon mucosa",
            "cancer-associated stroma",
            "colorectal adenocarcinoma epithelium",
        )

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
        # With early_stopping_rounds = 0
        # 01/11 04:34:58 [INFO]: training score: 0.998
        # 01/11 04:35:00 [INFO]: confusion matrix for training:
        # [[10406     1     0     0     0     0     0     0     0]
        # [    0 10565     0     0     0     0     0     1     0]
        # [    0     0 11480     0     3     1     0    25     3]
        # [    0     0     0 11557     0     0     0     0     0]
        # [    0     0     8     0  8875     4     0     4     5]
        # [    0     0    22     0    10 13481     0    23     0]
        # [    0     0     0     0     2     0  8757     1     3]
        # [    0     0    53     0     0    19     0 10368     6]
        # [    0     0     0     0     4     1     0     1 14311]]
        # 01/11 04:35:00 [INFO]: classification report for training:
        #                                     precision    recall  f1-score   support

        #                             adipose       1.00      1.00      1.00     10407
        #                         background       1.00      1.00      1.00     10566
        #                             debris       0.99      1.00      1.00     11512
        #                         lymphocytes       1.00      1.00      1.00     11557
        #                             mucus       1.00      1.00      1.00      8896
        #                     smooth muscle       1.00      1.00      1.00     13536
        #                 normal colon mucosa       1.00      1.00      1.00      8763
        #             cancer-associated stroma       0.99      0.99      0.99     10446
        # colorectal adenocarcinoma epithelium       1.00      1.00      1.00     14317

        #                             accuracy                           1.00    100000
        #                         macro avg       1.00      1.00      1.00    100000
        #                         weighted avg       1.00      1.00      1.00    100000

        # With early_stopping_rounds = 3
        # 01/11 05:27:24 [INFO]: training score: 0.97633
        # 01/11 05:27:26 [INFO]: confusion matrix for training:
        # [[10371     0     1     0    25     2     8     0     0]
        #  [   43 10492     0     0    13     0     2    10     6]
        #  [    0     3 11007    19    25   116    32   161   149]
        #  [    0     0     4 11468    13     1    32     8    31]
        #  [   14     2    23    12  8632    15   110    55    33]
        #  [    8     0    95     0    16 13243     7   157    10]
        #  [    3     0     4    23   133    18  8362    13   207]
        #  [    0     0   196     2    42   146    34  9979    47]
        #  [    0     1    25    22    31    30    60    69 14079]]
        # 01/11 05:27:26 [INFO]: classification report for training:
        #                                       precision    recall  f1-score   support

        #                              adipose       0.99      1.00      1.00     10407
        #                           background       1.00      0.99      1.00     10566
        #                               debris       0.97      0.96      0.96     11512
        #                          lymphocytes       0.99      0.99      0.99     11557
        #                                mucus       0.97      0.97      0.97      8896
        #                        smooth muscle       0.98      0.98      0.98     13536
        #                  normal colon mucosa       0.97      0.95      0.96      8763
        #             cancer-associated stroma       0.95      0.96      0.96     10446
        # colorectal adenocarcinoma epithelium       0.97      0.98      0.98     14317

        #                             accuracy                           0.98    100000
        #                            macro avg       0.98      0.98      0.98    100000
        #                         weighted avg       0.98      0.98      0.98    100000
        super().train()

    def test(self):
        # With early_stopping_rounds = 0
        # 01/11 04:33:41 [INFO]: testing score: 0.7034818941504178
        # 01/11 04:33:41 [INFO]: confusion matrix for testing:
        # [[1012    0    5    0   30  286    2    0    3]
        #  [   0  847    0    0    0    0    0    0    0]
        #  [   0    0  119    8    0  156    1   47    8]
        #  [   1    0    4  306  173    3   64    0   83]
        #  [  45   82    0   46  817    1   22    2   20]
        #  [   0    0  149    2    4  288   37   89   23]
        #  [   5    0   11   11   40   13  481    2  178]
        #  [   0    0  102    0    6   65   11  161   76]
        #  [   0    0   38   31   16    8  117    3 1020]]
        # 01/11 04:33:41 [INFO]: classification report for testing:
        #                                       precision    recall  f1-score   support

        #                              adipose       0.95      0.76      0.84      1338
        #                           background       0.91      1.00      0.95       847
        #                               debris       0.28      0.35      0.31       339
        #                          lymphocytes       0.76      0.48      0.59       634
        #                                mucus       0.75      0.79      0.77      1035
        #                        smooth muscle       0.35      0.49      0.41       592
        #                  normal colon mucosa       0.65      0.65      0.65       741
        #             cancer-associated stroma       0.53      0.38      0.44       421
        # colorectal adenocarcinoma epithelium       0.72      0.83      0.77      1233

        #                             accuracy                           0.70      7180
        #                            macro avg       0.66      0.64      0.64      7180
        #                         weighted avg       0.73      0.70      0.71      7180

        # With early_stopping_rounds = 3
        # 01/11 05:24:42 [INFO]: testing score: 0.7013927576601672
        # 01/11 05:24:42 [INFO]: confusion matrix for testing:
        # [[1016    0    5    0   20  286    5    0    6]
        # [   0  847    0    0    0    0    0    0    0]
        # [   0    0  119    8    0  156    1   51    4]
        # [   0    0    9  317  177    3   55    0   73]
        # [  47   93    1   43  804    2   26    2   17]
        # [   0    0  151    2    1  281   52   90   15]
        # [  10    0    6   27   30   15  469    1  183]
        # [   0    0   97    1   10   64   15  159   75]
        # [   0    0   42   23   12   14  117    1 1024]]
        # 01/11 05:24:42 [INFO]: classification report for testing:
        #                                     precision    recall  f1-score   support

        #                             adipose       0.95      0.76      0.84      1338
        #                         background       0.90      1.00      0.95       847
        #                             debris       0.28      0.35      0.31       339
        #                         lymphocytes       0.75      0.50      0.60       634
        #                             mucus       0.76      0.78      0.77      1035
        #                     smooth muscle       0.34      0.47      0.40       592
        #                 normal colon mucosa       0.63      0.63      0.63       741
        #             cancer-associated stroma       0.52      0.38      0.44       421
        # colorectal adenocarcinoma epithelium       0.73      0.83      0.78      1233

        #                             accuracy                           0.70      7180
        #                         macro avg       0.65      0.63      0.64      7180
        #                         weighted avg       0.72      0.70      0.71      7180
        super().test()
