#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s, MSG: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AMLS")
