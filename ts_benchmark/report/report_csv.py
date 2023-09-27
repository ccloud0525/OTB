# -*- coding: utf-8 -*-
import os

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.report.leader_board import get_leaderboard


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
    leaderboard_df = get_leaderboard(
        report_config["log_files_list"],
        report_config["aggregate_type"],
        report_config["report_metrics"],
        report_config["fill_type"],
        report_config["null_value_threshold"],
    )

    # Create final DataFrame and save to CSV
    leaderboard_df.to_csv(
        os.path.join(ROOT_PATH, "result", report_config["leaderboard_file_name"]),
        index=False,
    )
