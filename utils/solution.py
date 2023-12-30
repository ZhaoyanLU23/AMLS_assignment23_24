#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from typing import List
from abc import abstractmethod

from .dataset import Dataset
from .logger import logger


class Solution:
    def __init__(self, path: str, device: str):
        self.dataset = Dataset(path)
        self.task_name = "Task N"
        self.device = device

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

    def train(self):
        logger.info(f"[{self.task_name}] [Training] Running on {self.device}...")

    def test(self):
        logger.info(f"[{self.task_name}] [Testing] Running on {self.device}...")
