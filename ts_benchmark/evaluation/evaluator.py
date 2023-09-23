# -*- coding: utf-8 -*-
import functools
import traceback
from typing import List, Tuple, Any

import numpy as np


from ts_benchmark.evaluation.metrics import METRICS


class Evaluator:
    """
    评估器类，用于计算模型的评价指标。
    """

    METRICS_NAME = [
        "fit_time",
        "inference_time",
        "actual_data",
        "inference_data",
        "log",
    ]

    def __init__(self, metric: List[dict]):
        """
        初始化评估器对象。

        :param metric: 包含评价指标信息的列表。
        """
        self.metric = metric
        self.metric_funcs = []
        self.metric_names = []

        # 创建评价指标函数和名称列表
        for metric_info in self.metric:
            self.metric_names.append(metric_info.get("name"))
            metric_name_copy = metric_info.copy()
            name = metric_name_copy.pop("name")
            fun = METRICS[name]
            if metric_name_copy:
                self.metric_funcs.append(functools.partial(fun, **metric_name_copy))
            else:
                self.metric_funcs.append(fun)
        self.metric_names += self.METRICS_NAME
        self.n_metrics = len(self.metric_names)

    def evaluate(
        self, actual: np.ndarray, predicted: np.ndarray, hist_data: np.ndarray
    ) -> list:
        """
        计算模型的评价指标值。

        :param actual: 实际观测数据。
        :param predicted: 模型预测数据。
        :param hist_data: 历史数据（可选）。
        :return: 指标评价结果。
        """
        return [m(actual, predicted, hist_data=hist_data) for m in self.metric_funcs]

    def evaluate_with_log(
        self, actual: np.ndarray, predicted: np.ndarray, hist_data: np.ndarray
    ) -> Tuple[List[Any], str]:
        """
        计算模型的评价指标值。

        :param actual: 实际观测数据。
        :param predicted: 模型预测数据。
        :param hist_data: 历史数据（可选）。
        :return: 指标评价结果和log信息。
        """
        evaluate_result = []
        log_info = ""
        for m in self.metric_funcs:
            try:
                evaluate_result.append(m(actual, predicted, hist_data=hist_data)) 
            except Exception as e:
                evaluate_result.append(np.nan)
                log_info += (
                    f"Error in calculating {m.__name__}: {traceback.format_exc()}\n{e}\n"
                )
        return evaluate_result, log_info

    def default_result(self):
        """
        返回默认的评价指标结果。

        :return: 默认评价指标结果。
        """
        return self.n_metrics * [np.nan]
