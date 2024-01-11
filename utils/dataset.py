#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from .logger import logger

import numpy as np


class Dataset:
    """Load a MedMNIST dataset from its path and transform it into scikit-learn format."""

    def __init__(self, path: str) -> None:
        """Read data from path."""

        data = np.load(path)
        logger.info("=====================================")
        logger.info(f"     Loading {os.path.basename(path)}...")
        logger.info("=====================================")

        logger.info(
            "We concatenate the training set and validation set together for a better cross validation."
        )
        self.old_X_train = data.get("train_images", np.asarray([]))
        self.old_X_train = self._reshape_to_2_dims(self.old_X_train)
        X_val = data.get("val_images", np.asarray([]))
        self.X_val = self._reshape_to_2_dims(X_val)
        self.X_train = np.concatenate((self.old_X_train, self.X_val), axis=0)
        logger.debug(f"X_train shape: {self.X_train.shape}")

        self.old_y_train = data.get("train_labels", np.asarray([]))
        self.y_val = data.get("val_labels", np.asarray([]))
        self.y_train = np.concatenate((self.old_y_train, self.y_val), axis=0)
        logger.debug(f"y_train shape: {self.y_train.shape}")

        self.X_test = data.get("test_images", np.asarray([]))
        self.X_test = self._reshape_to_2_dims(self.X_test)
        logger.debug(f"X_test shape: {self.X_test.shape}")
        self.y_test = data.get("test_labels", np.asarray([]))
        logger.debug(f"y_test shape: {self.y_test.shape}")

        logger.info(f"Dataset loaded: {path}")

    def _reshape_to_2_dims(self, ndarray: np.ndarray) -> np.ndarray:
        from functools import reduce
        import operator

        n_samples = ndarray.shape[0]
        if n_samples > 0:
            n_features = reduce(operator.mul, ndarray.shape[1:])
            ndarray = ndarray.reshape((n_samples, n_features))
        return ndarray
