#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

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
        # 01/12 03:06:44 [INFO]: training score: 0.98143
        # 01/12 03:06:46 [INFO]: confusion matrix for training:
        # [[10379     0     1     0    20     1     6     0     0]
        #  [   31 10506     0     0    15     1     2     6     5]
        #  [    0     3 11094    22    18   106    33   127   109]
        #  [    0     0     2 11492     8     1    26     8    20]
        #  [   12     1    17    14  8671     7   101    56    17]
        #  [    5     0    62     1    11 13327     7   118     5]
        #  [    4     0     8    21   126    11  8434    12   147]
        #  [    0     0   111     2    37   116    30 10112    38]
        #  [    0     2    18    16    24    26    56    47 14128]]
        # 01/12 03:06:47 [INFO]: classification report for training:
        #                                       precision    recall  f1-score   support

        #                              adipose       1.00      1.00      1.00     10407
        #                           background       1.00      0.99      1.00     10566
        #                               debris       0.98      0.96      0.97     11512
        #                          lymphocytes       0.99      0.99      0.99     11557
        #                                mucus       0.97      0.97      0.97      8896
        #                        smooth muscle       0.98      0.98      0.98     13536
        #                 0.2,
        # {
        #                  normal colon mucosa       0.97      0.96      0.97      8763
        #             cancer-associated stroma       0.96      0.97      0.97     10446
        # colorectal adenocarcinoma epithelium       0.98      0.99      0.98     14317

        #                             accuracy                           0.98    100000
        #                            macro avg       0.98      0.98      0.98    100000
        #                         weighted avg       0.98      0.98      0.98    100000

        # With early_stopping_rounds = 3
        # 01/12 03:19:25 [INFO]: training score: 0.98118
        # 01/12 03:19:27 [INFO]: confusion matrix for training:
        # [[10379     0     1     0    20     1     6     0     0]
        #  [   33 10505     0     0    14     1     2     6     5]
        #  [    0     3 11089    22    19   103    34   133   109]
        #  [    0     0     2 11491     8     1    26     8    21]
        #  [   12     1    17    13  8674     7    98    54    20]
        #  [    5     0    64     1     9 13325     7   119     6]
        #  [    4     0     8    18   120    14  8435    13   151]
        #  [    0     0   109     2    37   120    30 10107    41]
        #  [    0     2    18    16    25    28    65    50 14113]]
        # 01/12 03:19:27 [INFO]: classification report for training:
        #                                       precision    recall  f1-score   support

        #                              adipose       0.99      1.00      1.00     10407
        #                           background       1.00      0.99      1.00     10566
        #                               debris       0.98      0.96      0.97     11512
        #                          lymphocytes       0.99      0.99      0.99     11557
        #                                mucus       0.97      0.98      0.97      8896
        #                        smooth muscle       0.98      0.98      0.98     13536
        #                  normal colon mucosa       0.97      0.96      0.97      8763
        #             cancer-associated stroma       0.96      0.97      0.97     10446
        # colorectal adenocarcinoma epithelium       0.98      0.99      0.98     14317

        #                             accuracy                           0.98    100000
        #                            macro avg       0.98      0.98      0.98    100000
        #                         weighted avg       0.98      0.98      0.98    100000
        super().train()

    def test(self):
        # With early_stopping_rounds = 0
        # 01/12 03:06:47 [INFO]: testing score: 0.7239554317548746
        # 01/12 03:06:47 [INFO]: confusion matrix for testing:
        # [[1044    4    7    0   18  256    2    0    7]
        #  [   0  847    0    0    0    0    0    0    0]
        #  [   0    0  127    8    0  156    1   44    3]
        #  [   0    3    7  364  134    2   72    0   52]
        #  [  32  144    0   28  794    1   23    1   12]
        #  [   0    0  134    7    1  298   46   92   14]
        #  [   5    1    4   23   20   16  520    0  152]
        #  [   0    0  112    1    3   66   14  160   65]
        #  [   0    0   48   24    8    6  102    1 1044]]
        # 01/12 03:06:47 [INFO]: classification report for testing:
        #                                       precision    recall  f1-score   support

        #                              adipose       0.97      0.78      0.86      1338
        #                           background       0.85      1.00      0.92       847
        #                               debris       0.29      0.37      0.33       339
        #                          lymphocytes       0.80      0.57      0.67       634
        #                                mucus       0.81      0.77      0.79      1035
        #                        smooth muscle       0.37      0.50      0.43       592
        #                  normal colon mucosa       0.67      0.70      0.68       741
        #             cancer-associated stroma       0.54      0.38      0.45       421
        # colorectal adenocarcinoma epithelium       0.77      0.85      0.81      1233

        #                             accuracy                           0.72      7180
        #                            macro avg       0.67      0.66      0.66      7180
        #                         weighted avg       0.75      0.72      0.73      7180

        # With early_stopping_rounds = 3
        # 01/12 02:51:16 [INFO]: [Task B] [Testing] Running on cuda:1...
        # 01/12 02:51:16 [INFO]: testing score: 0.7228412256267409
        # 01/12 02:51:16 [INFO]: confusion matrix for testing:
        # [[1043    5    4    0   20  255    4    0    7]
        #  [   0  847    0    0    0    0    0    0    0]
        #  [   0    0  124    8    0  158    1   44    4]
        #  [   0    2    4  357  135    3   73    0   60]
        #  [  31  137    0   28  801    1   22    1   14]
        #  [   0    0  135    6    2  296   45   92   16]
        #  [   6    1    7   21   25   13  515    0  153]
        #  [   0    0  103    1    3   67   12  166   69]
        #  [   0    0   47   24   10    8  103    0 1041]]
        # 01/12 02:51:16 [INFO]: classification report for testing:
        #                                       precision    recall  f1-score   support

        #                              adipose       0.97      0.78      0.86      1338
        #                           background       0.85      1.00      0.92       847
        #                               debris       0.29      0.37      0.33       339
        #                          lymphocytes       0.80      0.56      0.66       634
        #                                mucus       0.80      0.77      0.79      1035
        #                        smooth muscle       0.37      0.50      0.42       592
        #                  normal colon mucosa       0.66      0.70      0.68       741
        #             cancer-associated stroma       0.55      0.39      0.46       421
        # colorectal adenocarcinoma epithelium       0.76      0.84      0.80      1233

        #                             accuracy                           0.72      7180
        #                            macro avg       0.67      0.66      0.66      7180
        #                         weighted avg       0.74      0.72      0.73      7180
        super().test()
