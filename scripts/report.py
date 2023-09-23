# -*- coding: utf-8 -*-

from __future__ import absolute_import

import argparse
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

    report_dash.report(args.log_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparser = parser.add_subparsers(dest="method")

    add_dash_arguments(subparser.add_parser("dash"))

    args = parser.parse_args()

    {
        "dash": handle_dash_report
    }[args.method](args)

