#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging


class ColorfulFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s [%(levelname)s]: %(message)s"
    DATEFMT = "%m/%d %I:%M:%S"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.DATEFMT)
        return formatter.format(record)


logger = logging.getLogger("AMLS")

ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
formatter = ColorfulFormatter()
ch.setFormatter(formatter)

logger.addHandler(ch)


def set_log_level(level: int = logging.WARNING):
    global ch
    global logger
    ch.setLevel(level)
    logger.setLevel(level)
