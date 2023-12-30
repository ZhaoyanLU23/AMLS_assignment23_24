#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
from typing import List

from .dataset import Dataset
from .logger import logger


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
        params_config = val_config.get("params", {})
        logger.info(f"Searching xgboost params: {params_config}...")

    def train(self):
        logger.info(f"[{self.task_name}] [Training] Running on {self.device}...")

    def test(self):
        logger.info(f"[{self.task_name}] [Testing] Running on {self.device}...")
