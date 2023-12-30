#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .logger import logger

import numpy as np


class Dataset:
    """Load a MedMNIST dataset from its path and transform it into scikit-learn format."""

    def __init__(self, path: str) -> None:
        """Read data from path."""

        data = np.load(path)
        logger.info(f"Dataset[{path}] loading...")

        self.X_train = data.get("train_images", np.asarray([]))
        self.X_train = self._reshape_to_2_dims(self.X_train)
        logger.debug(f"X_train shape: {self.X_train.shape}")
        self.y_train = data.get("train_labels", np.asarray([]))
        logger.debug(f"y_train shape: {self.y_train.shape}")

        self.X_val = data.get("val_images", np.asarray([]))
        self.X_val = self._reshape_to_2_dims(self.X_val)
        logger.debug(f"X_val shape: {self.X_val.shape}")
        self.y_val = data.get("val_labels", np.asarray([]))
        logger.debug(f"y_val shape: {self.y_val.shape}")

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
