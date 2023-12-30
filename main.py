#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entrypoint of this assignment."""

import os
import sys
import logging

from utils.logger import logger
from A.solution import Solution as SolutionA
from B.solution import Solution as SolutionB

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
        help=f"task to solve: A, B, or all; default: all",
    )

    args, _ = parser.parse_known_args()
    return args


def print_info():
    print("-------------------------------------")
    print("|       AMLS Assignment 23-24       |")
    print("|         Name: Zhaoyan Lu          |")
    print("|        Student No: 23049710       |")
    print("-------------------------------------")


def solve(task: str):
    logger.info("-----------[Tasks running]-----------")
    if task in ["A", "all"]:
        task_a_data_abs_path = os.path.join(
            CWD, "Datasets", "PneumoniaMNIST", "pneumoniamnist.npz"
        )
        if os.path.exists(task_a_data_abs_path):
            solution_A = SolutionA()
            solution_A.solve(task_a_data_abs_path)
        else:
            raise Exception(f"No dataset for task A: {task_a_data_abs_path}!")

    if task in ["B", "all"]:
        task_b_data_abs_path = os.path.join(
            CWD, "Datasets", "PathMNIST", "pathmnist.npz"
        )
        if os.path.exists(task_b_data_abs_path):
            solution_B = SolutionB()
            solution_B.solve(task_b_data_abs_path)
        else:
            raise Exception(f"No dataset for task B: {task_b_data_abs_path}!")

    logger.info("-----------[Tasks finished]----------")


def main():
    try:
        args = setup_parse()
        logger.setLevel(args.loglevel)

        if args.action == "info":
            print_info()
        elif args.action == "solve":
            solve(args.task)
        else:
            raise Exception(f"Unsupported action: {args.action}")
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
