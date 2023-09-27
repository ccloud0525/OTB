# -*- coding: utf-8 -*-

from __future__ import absolute_import

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


def add_dash_arguments(parser):
    parser.add_argument(
        "-f",
        "--log-files",
        type=str,
        nargs="+",
        required=True,
        help="log file or directory path"
    )


def handle_dash_report(args):
    from ts_benchmark.report import report_dash

    report_dash.report({
        "log_files_list": args.log_files,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser = parser.add_subparsers(dest="method")

    add_dash_arguments(subparser.add_parser("dash"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parser.parse_args()

    {
        "dash": handle_dash_report
    }[args.method](args)

