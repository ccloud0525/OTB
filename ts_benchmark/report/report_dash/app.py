# -*- coding: utf-8 -*-

from __future__ import absolute_import

import itertools
import logging
import os
from typing import Union, List

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from flask import Flask, redirect

from ts_benchmark.report.report_dash.memory import READONLY_MEMORY

logger = logging.getLogger(__name__)


def _find_log_files(directory: str) -> List[str]:
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


def _load_log_data(log_files: List[str]) -> pd.DataFrame:
    log_files = itertools.chain.from_iterable([
        [fn] if not os.path.isdir(fn) else _find_log_files(fn) for fn in log_files
    ])

    ret = []
    for fn in log_files:
        ret.append(pd.read_csv(fn))
    return pd.concat(ret, axis=0)


def report(log_files: Union[List[str], pd.DataFrame]):

    # currently we do not support showing artifact columns
    artifact_columns = ["actual_data", "inference_data", "log"]

    log_data = log_files if isinstance(log_files, pd.DataFrame) else _load_log_data(log_files)
    log_data = log_data.drop(columns=artifact_columns)

    READONLY_MEMORY["raw_data"] = log_data

    server = Flask(__name__)

    @server.route('/')
    def index_redirect():
        return redirect('/leaderboard')

    app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)

    # Define the layout
    app.layout = html.Div([
        dash.page_container
    ])

    app.run(host="0.0.0.0", debug=True)
