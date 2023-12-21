#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entrypoint of this assignment."""

import sys

from A.solution import Solution as SolutionA
from B.solution import Solution as SolutionB


def setup_parse():
    import argparse

    description = "AMLS Final Assignment"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", dest="verbose", action="store_true", help="verbose")
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
    print("-----------[Tasks running]-----------")
    if task in ["A", "all"]:
        solution_A = SolutionA()
        solution_A.solve()

    if task in ["B", "all"]:
        solution_B = SolutionB()
        solution_B.solve()
    print("-----------[Tasks finished]----------")


def main():
    try:
        args = setup_parse()

        if args.action == "info":
            print_info()
        elif args.action == "solve":
            solve(args.task)
        else:
            raise Exception(f"Unsupport action: {args.action}")
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
