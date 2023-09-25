# -*- coding: utf-8 -*-
import base64
import pickle
import time
import traceback
from typing import List, Any

import pandas as pd

from ts_benchmark.data_loader.data_pool import DataPool
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import classification_metrics_label
from ts_benchmark.evaluation.metrics import classification_metrics_score
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.strategy import Strategy
from ts_benchmark.models.get_model import ModelFactory
from ts_benchmark.utils.data_processing import split_before


class AnomalyDetect(Strategy):
    """
    异常检测类，用于在时间序列数据上执行异常检测。
    """

    def __init__(self, strategy_config: dict, evaluator: Evaluator):
        """
        初始化子类实例。

        :param strategy_config: 模型评估配置。
        """
        super().__init__(strategy_config, evaluator)
        self.model = None
        self.data_lens = None

    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        """
        执行异常检测策略。

        :param series_name: 要执行异常检测的序列名称。
        :param model_factory: 模型对象的构造/工厂函数。
        :return: 评估结果。
        """
        model = model_factory()
        try:
            self.model = model
            train_data, train_label, test_data, test_label = self.split_data(
                series_name
            )
            start_fit_time = time.time()
            if hasattr(model, "detect_fit"):
                self.model.detect_fit(train_data, train_label)  # 在训练数据上拟合模型
            else:
                self.model.fit(train_data, train_label)  # 在训练数据上拟合模型
            end_fit_time = time.time()
            predict_label = self.detect(test_data)
            end_inference_time = time.time()
            actual_label = test_label.to_numpy().flatten()
            single_series_results, log_info = self.evaluator.evaluate_with_log(
                actual_label.astype(float), predict_label.astype(float), train_data.values
            )
            inference_data = pd.DataFrame(
                predict_label, columns=test_label.columns, index=test_label.index
            )
            actual_data_pickle = pickle.dumps(test_label)
            # 使用 base64 进行编码
            actual_data_pickle = base64.b64encode(actual_data_pickle).decode(
                "utf-8"
            )

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
            single_series_results = self.get_default_result(**{FieldNames.LOG_INFO: log})
        return single_series_results

    def split_data(self, data: str):
        raise NotImplementedError

    def detect(self, test_data: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def accepted_metrics():
        raise NotImplementedError

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


class FixedDetectScore(AnomalyDetect):
    REQUIRED_FIELDS = ["train_test_split"]

    def split_data(self, series_name):
        data = DataPool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(
            self.strategy_config["train_test_split"] * self.data_lens
        )
        train, test = split_before(data, train_length)
        train_data, train_label = train.loc[:, train.columns != 'label'], train.loc[:, ['label']]
        test_data, test_label = test.loc[:, train.columns != 'label'], test.loc[:, ['label']]
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class FixedDetectLabel(AnomalyDetect):
    REQUIRED_FIELDS = ["train_test_split"]

    def split_data(self, series_name: str):
        data = DataPool().get_series(series_name)
        self.data_lens = len(data)
        train_length = int(
            self.strategy_config["train_test_split"] * self.data_lens
        )
        train, test = split_before(data, train_length)
        train_data, train_label = train.loc[:, train.columns != 'label'], train.loc[:, ['label']]
        test_data, test_label = test.loc[:, train.columns != 'label'], test.loc[:, ['label']]
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class UnFixedDetectScore(AnomalyDetect):
    def split_data(self, series_name: str):
        data = DataPool().get_series(series_name)
        train_length = int(DataPool().get_series_meta_info(series_name)["train_lens"].item())
        train, test = split_before(data, train_length)
        train_data, train_label = train.loc[:, train.columns != 'label'], train.loc[:, ['label']]

        test_data, test_label = test.loc[:, train.columns != 'label'], test.loc[:, ['label']]
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class UnFixedDetectLabel(AnomalyDetect):
    def split_data(self, series_name):
        data = DataPool().get_series(series_name)
        train_length = int(DataPool().get_series_meta_info(series_name)["train_lens"].item())
        train, test = split_before(data, train_length)
        train_data, train_label = train.loc[:, train.columns != 'label'], train.loc[:, ['label']]
        test_data, test_label = test.loc[:, train.columns != 'label'], test.loc[:, ['label']]
        return train_data, train_label, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__


class AllDetectScore(AnomalyDetect):
    def split_data(self, series_name):
        data = DataPool().get_series(series_name)
        train = data
        test = data
        train_data, train_label = train.loc[:, train.columns != 'label'], None
        test_data, test_label = test.loc[:, train.columns != 'label'], test.loc[:, ['label']]
        return train_data, None, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_score(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_score.__all__


class AllDetectLabel(AnomalyDetect):
    def split_data(self, series_name):
        data = DataPool().get_series(series_name)
        train = data
        test = data
        train_data, train_label = train.loc[:, train.columns != 'label'], None
        test_data, test_label = test.loc[:, train.columns != 'label'], test.loc[:, ['label']]
        return train_data, None, test_data, test_label

    def detect(self, test_data):
        return self.model.detect_label(test_data)

    @staticmethod
    def accepted_metrics():
        return classification_metrics_label.__all__
