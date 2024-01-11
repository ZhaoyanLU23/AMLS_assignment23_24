#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entrypoint of this assignment."""

import os
import sys
import logging
from typing import List

from constants import (
    TASK_A_DATA_ABS_PATH,
    TASK_B_DATA_ABS_PATH,
    TASK_A_CONFIG_PATH,
    TASK_B_CONFIG_PATH,
)
from utils.logger import logger, set_log_level
from A.solution import SolutionA
from B.solution import SolutionB

CWD = os.getcwd()


def setup_parse():
    import argparse

    description = "AMLS Final Assignment"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    parser.set_defaults(whether_output=True)

    subparsers = parser.add_subparsers(dest="action", help="actions provided")
    subparsers.required = True

    solve_subparser = subparsers.add_parser("solve")
    subparsers.add_parser("info")

    solve_subparser.add_argument(
        "--task",
        action="store",
        default="all",
        help="task to solve: A, B, or all; default: all",
    )
    solve_subparser.add_argument(
        "--stages",
        action="store",
        default="val,train,test",
        help="task stages: val, train, test, or all; default: val,train,test",
    )
    solve_subparser.add_argument(
        "--device",
        action="store",
        default="cuda",
        help="Device ordinal for xgboost, available options: cpu, cuda, and gpu; default: cuda",
    )
    solve_subparser.add_argument(
        "--save",
        action="store",
        default="True",
        help="Save results or not, available options: True or False; default: True",
    )
    solve_subparser.add_argument(
        "--early_stopping_rounds",
        action="store",
        default="3",
        help="Used to prevent overfitting; require >= 0, default: 3",
    )

    args, _ = parser.parse_known_args()
    return args


def print_info():
    print("-------------------------------------")
    print("|       AMLS Assignment 23-24       |")
    print("|         Name: Zhaoyan Lu          |")
    print("|        Student No: 23049710       |")
    print("-------------------------------------")


def solve(
    task: str,
    stages: List[str],
    device: str,
    early_stopping_rounds: int,
    save: bool = True,
):
    logger.info("-----------[Tasks running]-----------")
    if task in ["A", "all"]:
        if os.path.exists(TASK_A_DATA_ABS_PATH):
            solution_A = SolutionA(
                dataset_path=TASK_A_DATA_ABS_PATH,
                device=device,
                config_path=TASK_A_CONFIG_PATH,
                save_result=save,
                early_stopping_rounds=early_stopping_rounds,
            )
            solution_A.solve(stages)
        else:
            raise Exception(
                f"No dataset for task A: {TASK_A_DATA_ABS_PATH}! Please run `make download`."
            )

    if task in ["B", "all"]:
        if os.path.exists(TASK_B_DATA_ABS_PATH):
            solution_B = SolutionB(
                dataset_path=TASK_B_DATA_ABS_PATH,
                device=device,
                config_path=TASK_B_CONFIG_PATH,
                save_result=save,
                early_stopping_rounds=early_stopping_rounds,
            )
            solution_B.solve(stages)
        else:
            raise Exception(
                f"No dataset for task B: {TASK_B_DATA_ABS_PATH}! Please run `make download`."
            )

    logger.info("---------All tasks finished!---------")


def main():
    try:
        args = setup_parse()
        set_log_level(args.loglevel)

        if args.action == "info":
            print_info()
        elif args.action == "solve":
            stages = args.stages.split(",")
            save_result = args.save.lower() == "true"
            assert (
                args.early_stopping_rounds.isdigit()
                and int(args.early_stopping_rounds) >= 0
            ), f"ERROR: early_stopping_rounds: '{args.early_stopping_rounds}' needs to be greater than or equal to 0."
            early_stopping_rounds = int(args.early_stopping_rounds)
            solve(args.task, stages, args.device, early_stopping_rounds, save_result)
        else:
            raise Exception(f"Unsupported action: {args.action}")
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
