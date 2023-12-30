#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from typing import List
from abc import abstractmethod

from .dataset import Dataset
from .logger import logger


class Solution:
    def __init__(self, path: str):
        self.dataset = Dataset(path)
        self.task_name = "Task N"

    def solve(self, stages: List[str], device: str):
        logger.info("=====================================")
        logger.info(f"|         Solving {self.task_name} ...        |")
        logger.info("=====================================")

        if "val" in stages:
            self.val()
        if "train" in stages:
            self.train(device=device)
        if "test" in stages:
            self.test()

        logger.info(f"----------{self.task_name} finished!-----------")

    @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def train(self, device: str):
        pass

    @abstractmethod
    def test(self):
        pass
