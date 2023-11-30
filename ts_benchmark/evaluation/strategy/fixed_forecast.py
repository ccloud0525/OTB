# -*- coding: utf-8 -*-
import base64
import pickle
import time
import traceback
from typing import Any, List
import pandas as pd

from ts_benchmark.data_loader.data_pool import DataPool
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import regression_metrics
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.strategy import Strategy
from ts_benchmark.models.get_model import ModelFactory
from ts_benchmark.utils.data_processing import split_before
from ts_benchmark.utils.random_utils import fix_random_seed


class FixedForecast(Strategy):
    """
    固定预测策略类，用于在时间序列数据上执行固定预测。
    """

    REQUIRED_FIELDS = ["pred_len"]

    def __init__(self, strategy_config: dict, evaluator: Evaluator):
        """
        初始化固定预测策略对象。
        :param strategy_config: 模型评估配置。
        """
        super().__init__(strategy_config, evaluator)
        self.pred_len = self.strategy_config["pred_len"]

    # def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
    #     """
    #     执行固定预测策略。
    #
    #     :param series_name: 要执行预测的序列名称。
    #     :param model_factory: 模型对象的构造/工厂函数。
    #     :return: 评估结果。
    #     """
    #     fix_random_seed()
    #     model = model_factory()
    #     data = DataPool().get_series(series_name)
    #     try:
    #         train_length = len(data) - self.pred_len
    #         if train_length <= 0:
    #             raise ValueError("The prediction step exceeds the data length")
    #         train, test = split_before(data, train_length)  # 分割训练和测试数据
    #
    #         self.scaler.fit(train.values)
    #
    #         train_data = pd.DataFrame(self.scaler.transform(train.values), columns=train.columns, index=train.index)
    #
    #         start_fit_time = time.time()
    #         if hasattr(model, "forecast_fit"):
    #             model.forecast_fit(train_data)  # 在训练数据上拟合模型
    #         else:
    #             model.fit(train_data)  # 在训练数据上拟合模型
    #         end_fit_time = time.time()
    #         predict = model.forecast(self.pred_len, train_data)  # 预测未来数据
    #
    #         predict = self.scaler.inverse_transform(predict)
    #         end_inference_time = time.time()
    #
    #         actual = test.to_numpy()
    #
    #         single_series_results, log_info = self.evaluator.evaluate_with_log(
    #             actual, predict, train.values
    #         )  # 计算评价指标
    #
    #         inference_data = pd.DataFrame(
    #             predict, columns=test.columns, index=test.index
    #         )
    #         actual_data_pickle = pickle.dumps(test)
    #         # 使用 base64 进行编码
    #         actual_data_pickle = base64.b64encode(actual_data_pickle).decode("utf-8")
    #
    #         inference_data_pickle = pickle.dumps(inference_data)
    #         # 使用 base64 进行编码
    #         inference_data_pickle = base64.b64encode(inference_data_pickle).decode(
    #             "utf-8"
    #         )
    #
    #         single_series_results += [
    #             series_name,
    #             end_fit_time - start_fit_time,
    #             end_inference_time - end_fit_time,
    #             actual_data_pickle,
    #             inference_data_pickle,
    #             log_info,
    #         ]
    #
    #     except Exception as e:
    #         log = f"{traceback.format_exc()}\n{e}"
    #         single_series_results = self.get_default_result(
    #             **{FieldNames.LOG_INFO: log}
    #         )
    #
    #     return single_series_results
    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        """
        执行固定预测策略。

        :param series_name: 要执行预测的序列名称。
        :param model_factory: 模型对象的构造/工厂函数。
        :return: 评估结果。
        """

        print(series_name)
        fix_random_seed()
        model = model_factory()
        data = DataPool().get_series(series_name)
        try:
            train_length = len(data) - self.pred_len
            if train_length <= 0:
                raise ValueError("The prediction step exceeds the data length")
            train, test = split_before(data, train_length)  # 分割训练和测试数据

            # ----------------------------------------------------------------------------------可以删除
            # train_data1, rest1 = split_before(train, int(train_length * 0.875))
            # self.scaler.fit(train_data1.values)
            self.scaler.fit(train.values)
            # ----------------------------------------------------------------------------------可以删除

            start_fit_time = time.time()
            if hasattr(model, "forecast_fit"):
                model.forecast_fit(train, 0.875)  # 在训练数据上拟合模型
            else:
                model.fit(train, 0.875)  # 在训练数据上拟合模型
            end_fit_time = time.time()
            predict = model.forecast(self.pred_len, train)  # 预测未来数据

            end_inference_time = time.time()

            actual = test.to_numpy()

            single_series_results, log_info = self.evaluator.evaluate_with_log(
                actual, predict, train.values
            )  # 计算评价指标

            # ----------------------------------------------------------------------------------可以删除
            transformed_predict = self.scaler.transform(predict)
            transformed_actual = pd.DataFrame(
                self.scaler.transform(test.values),
                columns=test.columns,
                index=test.index,
            ).to_numpy()

            transformed_single_series_result = self.evaluator.evaluate(  # 计算评价指标
                transformed_actual, transformed_predict, train.values
            )


            # 对于单变量而言不需要归一化
            transformed_single_series_result = single_series_results
            # 对于单变量而言不需要归一化

            single_series_results = [
                str(f"{a};{b}")

                for a, b in zip(
                    single_series_results, transformed_single_series_result
                )
            ]
            # ----------------------------------------------------------------------------------可以删除

            inference_data = pd.DataFrame(
                predict, columns=test.columns, index=test.index
            )
            actual_data_pickle = pickle.dumps(test)
            # 使用 base64 进行编码
            actual_data_pickle = base64.b64encode(actual_data_pickle).decode("utf-8")

            inference_data_pickle = pickle.dumps(inference_data)
            # 使用 base64 进行编码
            inference_data_pickle = base64.b64encode(inference_data_pickle).decode(
                "utf-8"
            )

            single_series_results += [
                series_name,
                end_fit_time - start_fit_time,
                end_inference_time - end_fit_time,
                actual_data_pickle,
                inference_data_pickle,
                log_info,
            ]

        except Exception as e:
            log = f"{traceback.format_exc()}\n{e}"
            single_series_results = self.get_default_result(
                **{FieldNames.LOG_INFO: log}
            )

        return single_series_results
    @staticmethod
    def accepted_metrics():
        """
        获取固定预测策略接受的评价指标列表。

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
