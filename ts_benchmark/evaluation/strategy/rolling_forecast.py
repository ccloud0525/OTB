# -*- coding: utf-8 -*-
import base64
import pickle
import time
import traceback
from typing import List, Any

import numpy as np
import pandas as pd

from ts_benchmark.data_loader.data_pool import DataPool
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import regression_metrics
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.strategy import Strategy
from ts_benchmark.models.get_model import ModelFactory
from ts_benchmark.utils.data_processing import split_before


class RollingForecast(Strategy):
    REQUIRED_FIELDS = ["pred_len", "train_test_split", "stride", "num_rollings"]

    """
    滚动预测策略类，用于在时间序列数据上执行滚动预测。
    """

    def __init__(self, strategy_config: dict, evaluator: Evaluator):
        """
        初始化滚动预测策略对象。


        :param strategy_config: 模型评估配置。
        """
        super().__init__(strategy_config, evaluator)
        self.data_lens = None
        self.pred_len = self.strategy_config["pred_len"]
        self.num_rollings = self.strategy_config["num_rollings"]

    def _get_index(self, test_length: int, train_length: int) -> List[int]:
        """
        获取滚动窗口的索引列表。

        :param test_length: 测试数据长度。
        :param train_length: 训练数据长度。
        :return: 滚动窗口的索引列表。
        """
        stride = self.strategy_config["stride"]
        index_list = list(
            range(train_length, self.data_lens - self.pred_len + 1, stride)
        ) + (
            [self.data_lens - self.pred_len]
            if (test_length - self.pred_len) % stride != 0
            else []
        )
        return index_list

    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        """
        执行滚动预测策略。

        :param series_name: 要执行预测的序列名称。
        :param model_factory: 模型对象的构造/工厂函数。
        :return: 评估结果的平均值。
        """
        model = model_factory()

        data = DataPool().get_series(series_name)
        self.data_lens = len(data)
        try:
            all_test_results = []

            train_length = int(
                self.strategy_config["train_test_split"]
                * self.data_lens
            )
            test_length = self.data_lens - train_length
            if train_length <= 0 or test_length <= 0:
                raise ValueError(
                    "The length of training or testing data is less than or equal to 0"
                )
            train_data, other = split_before(data, train_length)  # 分割训练数据
            start_fit_time = time.time()
            if hasattr(model, "forecast_fit"):
                model.forecast_fit(train_data)  # 在训练数据上拟合模型
            else:
                model.fit(train_data)  # 在训练数据上拟合模型
            end_fit_time = time.time()
            index_list = self._get_index(test_length, train_length)  # 获取滚动窗口的索引列表
            total_inference_time = 0
            all_rolling_actual = []
            all_rolling_predict = []
            for i in range(min(len(index_list), self.num_rollings)):
                index = index_list[i]
                train, other = split_before(data, index)  # 分割训练数据
                test, rest = split_before(other, self.pred_len)  # 分割测试数据
                start_inference_time = time.time()
                predict = model.forecast(self.pred_len, train)  # 预测未来数据
                end_inference_time = time.time()
                total_inference_time += end_inference_time - start_inference_time
                actual = test.to_numpy()

                single_series_result = self.evaluator.evaluate(  # 计算评价指标
                    actual, predict, train_data.values
                )
                inference_data = pd.DataFrame(
                    predict, columns=test.columns, index=test.index
                )

                all_rolling_actual.append(test)
                all_rolling_predict.append(inference_data)
                all_test_results.append(single_series_result)
            average_inference_time = float(total_inference_time) / min(
                len(index_list), self.num_rollings
            )
            single_series_results = np.mean(
                np.stack(all_test_results), axis=0
            ).tolist()  # 对所有滚动结果求均值

            all_rolling_actual_pickle = pickle.dumps(all_rolling_actual)
            # 使用 base64 进行编码
            all_rolling_actual_pickle = base64.b64encode(all_rolling_actual_pickle).decode(
                "utf-8"
            )

            all_rolling_predict_pickle = pickle.dumps(all_rolling_predict)
            # 使用 base64 进行编码
            all_rolling_predict_pickle = base64.b64encode(all_rolling_predict_pickle).decode(
                "utf-8"
            )

            single_series_results += [
                series_name,
                end_fit_time - start_fit_time,
                average_inference_time,
                all_rolling_actual_pickle,
                all_rolling_predict_pickle,
                "",
            ]
        except Exception as e:
            log = f"{traceback.format_exc()}\n{e}"
            single_series_results = self.get_default_result(**{FieldNames.LOG_INFO: log})

        return single_series_results

    @staticmethod
    def accepted_metrics() -> List[str]:
        """
        获取滚动预测策略接受的评价指标列表。

        :return: 评价指标列表。
        """
        return regression_metrics.__all__  # 返回评价指标列表

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]
