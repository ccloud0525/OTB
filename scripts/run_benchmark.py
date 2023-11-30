# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import random
import sys
import warnings

import numpy as np
import torch


sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../ts_benchmark/baselines/third_party")
)
from ts_benchmark.utils.get_file_name import get_log_file_name
from ts_benchmark.report import report_dash, report_csv
from ts_benchmark.common.constant import CONFIG_PATH
from ts_benchmark.pipeline import pipeline
from ts_benchmark.utils.parallel import ParallelBackend
from ts_benchmark.utils.random_utils import fix_random_seed

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run_benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # fix_random_seed()

    # script name
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        choices=[
            "fixed_forecast_config.json",
            "rolling_forecast_config.json",
            "fixed_detect_score_config.json",
            "fixed_detect_label_config.json",
            "unfixed_detect_score_config.json",
            "unfixed_detect_label_config.json",
            "all_detect_score_config.json",
            "all_detect_label_config.json",

            "fixed_forecast_config_daily.json",
            "fixed_forecast_config_monthly.json",
            "fixed_forecast_config_other.json",
            "fixed_forecast_config_quarterly.json",
            "fixed_forecast_config_weekly.json",
            "fixed_forecast_config_yearly.json",
            "fixed_forecast_config_hourly.json"

        ],
        help="evaluation config file path",
    )
    # data_loader_config
    parser.add_argument(
        "--data-set-name",
        type=str,
        required=True,
        choices=[
            "large_forecast",
            "medium_forecast",
            "small_forecast",
            "large_detect",
            "medium_detect",
            "small_detect",
        ],
        help="dataset name",
    )

    parser.add_argument(
        "--typical-data-name-list",
        type=str,
        nargs="+",
        default=None,
        help="typical_data_name_list",
    )

    # model_config
    parser.add_argument(
        "--adapter",
        type=str,
        nargs="+",
        default=None,
        help="Adapters for converting models",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        nargs="+",
        required=True,
        help="model path to evaluate",
    )
    parser.add_argument(
        "--model-hyper-params",
        type=str,
        nargs="+",
        default=None,
        help=(
            "The input parameters corresponding to the models to be evaluated "
            "should correspond one-to-one with the --model-name options."
        ),
    )

    # model_eval_config
    parser.add_argument(
        "--metric-name",
        type=str,
        nargs="+",
        default="all",
        help="metrics to be evaluated",
    )

    parser.add_argument(
        "--strategy-args",
        type=str,
        default=None,
        help="Parameters required for evaluating strategies",
    )

    # evaluation engine
    parser.add_argument(
        "--eval-backend",
        type=str,
        default="sequential",
        choices=["sequential", "ray"],
        help="evaluation backend, use ray for parallel evaluation",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=os.cpu_count(),
        help="number of cpus to use, only available in certain backends",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="list of gpu devices to use, only available in certain backends",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
        help="number of evaluation workers",
    )
    # TODO: should timeout be part of the configuration file?
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="time limit for each evaluation task, in seconds",
    )

    # report_config
    parser.add_argument(
        "--aggregate_type",
        default="mean",
        help="Select the baseline algorithm to compare",
    )

    parser.add_argument(
        "--report-method",
        type=str,
        default="csv",
        choices=[
            "dash",
            "csv",
        ],
        help="Presentation form of algorithm performance comparison results",
    )


    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )




    torch.set_num_threads(3)

    with open(os.path.join(CONFIG_PATH, args.config_path), "r") as file:
        config_data = json.load(file)

    required_configs = [
        "data_loader_config",
        "model_config",
        "model_eval_config",
        "report_config",
    ]
    for config_name in required_configs:
        config = config_data.get(config_name, None)
        if config is None:
            raise ValueError(f"{config_name} is none")

    data_loader_config = config_data["data_loader_config"]
    data_loader_config["data_set_name"] = args.data_set_name
    data_loader_config["typical_data_name_list"] = args.typical_data_name_list

    model_config = config_data.get("model_config", None)

    if args.adapter is not None:
        if len(args.model_name) > len(args.adapter):
            args.adapter.extend([None] * (len(args.model_name) - len(args.adapter)))

    if args.model_hyper_params is not None:
        if len(args.model_name) > len(args.model_hyper_params):
            args.model_hyper_params.extend([None] * (len(args.model_name) - len(args.model_hyper_params)))

    args.adapter = (
        [None if item == "None" else item for item in args.adapter]
        if args.adapter is not None
        else [None] * len(args.model_name)
    )

    args.model_hyper_params = (
        [None if item == "None" else item for item in args.model_hyper_params]
        if args.model_hyper_params is not None
        else [None] * len(args.model_name)
    )

    for adapter, model_name, model_hyper_params in zip(
        args.adapter, args.model_name, args.model_hyper_params
    ):
        model_config["models"].append(
            {
                "adapter": adapter if adapter is not None else None,
                "model_name": model_name,
                "model_hyper_params": json.loads(model_hyper_params)
                if model_hyper_params is not None
                else {},
            }
        )



    model_eval_config = config_data["model_eval_config"]

    metric_list = []
    if args.metric_name != 'all':
        for metric in args.metric_name:
            metric_name = json.loads(metric)
            metric_list.append(metric_name)
        model_eval_config["metric_name"] = metric_list
    else:
         model_eval_config["metric_name"] = args.metric_name


    default_strategy_args = model_eval_config["strategy_args"]
    specific_strategy_args = (
        json.loads(args.strategy_args) if args.strategy_args else None
    )

    if specific_strategy_args is not None:
        default_strategy_args.update(specific_strategy_args)
        model_eval_config["strategy_args"] = default_strategy_args

    report_config = config_data["report_config"]
    report_config["aggregate_type"] = args.aggregate_type
    report_config["saved_path"] = args.saved_path

    ParallelBackend().init(
        backend=args.eval_backend,
        n_workers=args.num_workers,
        n_cpus=args.num_cpus,
        gpu_devices=args.gpus,
        default_timeout=args.timeout,
        max_tasks_per_child=5,
    )
    # try:
    #     log_filenames = pipeline(data_loader_config, model_config, model_eval_config)
    # finally:
    #     ParallelBackend().close(force=True)
    #
    # report_config["log_files_list"] = log_filenames
    # if args.report_method == "dash":
    #     report_dash.report(report_config)
    # elif args.report_method == "csv":
    #     # report_config["leaderboard_file_name"] = "test_report.csv"
    #     filename = get_log_file_name()
    #     leaderboard_file_name = "test_report" + filename
    #     # report_config["leaderboard_file_name"] = "test_report.csv"
    #     report_config["leaderboard_file_name"] = leaderboard_file_name
    #     report_csv.report(report_config)
    try:
        log_filenames = pipeline(data_loader_config, model_config, model_eval_config, report_config["saved_path"])

    finally:
        ParallelBackend().close(force=True)

    report_config["log_files_list"] = log_filenames
    if args.report_method == "dash":
        report_dash.report(report_config)
    elif args.report_method == "csv":
        # report_config["leaderboard_file_name"] = "test_report.csv"
        filename = get_log_file_name()
        leaderboard_file_name = "test_report" + filename
        # report_config["leaderboard_file_name"] = "test_report.csv"
        report_config["leaderboard_file_name"] = leaderboard_file_name

        report_csv.report(report_config)

