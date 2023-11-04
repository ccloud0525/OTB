# -*- coding: utf-8 -*-

import os
from typing import Union, List

import pandas as pd

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.report.leaderboard import get_leaderboard
from ts_benchmark.report.report_dash.app import _load_log_data


def report(report_config: dict) -> None:
    """
    Generate a report based on specified configuration parameters.

    Parameters:
    - report_config (dict): A dictionary containing the following keys and their respective values:
        - log_files_list (List[str]): A list of file paths for log files.
        - leaderboard_file_name (str): The name for the saved report file.
        - aggregate_type (str): The aggregation type used when reporting the final results of evaluation metrics.
        - report_metrics (Union[str, List[str]]): The metrics for the report, can be a string or a list of strings.
        - fill_type (str): The type of fill for missing values.
        - null_value_threshold (float): The threshold value for null metrics.

    Raises:
    - ValueError: If all metrics have too many null values, making performance comparison impossible.

    Returns:
    - None: The function does not return a value, but generates and saves a report to a CSV file.
    """
    log_files: Union[List[str], pd.DataFrame] = report_config.get("log_files_list")
    if not log_files:
        raise ValueError("No log files to report")

    log_data = (
        log_files if isinstance(log_files, pd.DataFrame) else _load_log_data(log_files)
    )

    # ---------------------------------------------------------删除
    selected_column = report_config["report_metrics"]  # 替换为您想要处理的列的名称
    for column in selected_column:
        column_index = log_data.columns.get_loc(column)
        for row in range(log_data.shape[0]):
            log_data.iloc[row, column_index] = float(log_data.iloc[row, column_index].split(';')[0])
    # ---------------------------------------------------------删除


    leaderboard_df = get_leaderboard(
        log_files,
        log_data,
        report_config.get("aggregate_type", "mean"),
        report_config["report_metrics"],
        report_config.get("fill_type", "mean_value"),
        report_config.get("null_value_threshold", 0.3),
    )

    num_rows = leaderboard_df.shape[0]
    leaderboard_df.insert(0, 'strategy_args', [log_data.iloc[0, 1]] * num_rows)

    # Create final DataFrame and save to CSV
    leaderboard_df.to_csv(
        os.path.join(ROOT_PATH, "result", report_config["leaderboard_file_name"]),
        index=False,
    )
