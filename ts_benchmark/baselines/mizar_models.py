# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from mizar.models.forecasting import ARIMA
from mizar.models.detection import InterQuartileRangeAD
from mizar.models.detection.sigmaad import StreamSigmaAD


class MizarModelAdapter:
    """
    Darts 模型适配器类，用于适配 Darts 框架中的模型，使其符合预测策略的需求。
    """

    def __init__(
        self,
        model_name: str,
        model_class: object,
        model_args: dict,
        allow_fit_on_eval: bool,
    ):
        """
        初始化 Darts 模型适配器对象。

        :param model_name: 模型名称。
        :param model_class: Darts 模型类。
        :param model_args: 模型初始化参数。
        :param allow_fit_on_eval: 是否允许在预测阶段拟合模型。
        """
        self.model = None
        self.model_class = model_class
        self.model_args = model_args
        self.model_name = model_name
        self.allow_fit_on_eval = allow_fit_on_eval

    def fit(self, series: pd.DataFrame) -> object:
        """
        在时间序列数据上拟合适配的 Darts 模型。

        :param series: 时间序列数据。
        :return: 拟合后的模型对象。
        """
        self.model = self.model_class(self.model_args)
        return self.model.fit(series)

    def forecast(self, pred_len: int, train: pd.DataFrame) -> np.ndarray:
        """
        使用适配的 Darts 模型进行预测。

        :param pred_len: 预测长度。
        :param train: 用于拟合模型的训练数据。
        :return: 预测结果。
        """
        if self.allow_fit_on_eval:
            self.fit(train)
            fsct_result = self.model.forecast(pred_len)
        else:
            fsct_result = self.model.forecast(pred_len, train)
        predict = fsct_result["mean"].values
        return predict

    def detect(self, test: pd.DataFrame) -> np.ndarray:
        detect_result = self.model.detect(test)
        return detect_result

    def __repr__(self):
        """
        返回模型名称的字符串表示。
        """
        return self.model_name


def generate_model_factory(
    model_name: str, model_class: object, required_args: dict, allow_fit_on_eval: bool
) -> object:
    """
    生成模型工厂信息，用于创建 Darts 模型适配器。

    :param model_name: 模型名称。
    :param model_class: Darts 模型类。
    :param required_args: 模型初始化所需参数。
    :param allow_fit_on_eval: 是否允许在预测阶段拟合模型。
    :return: 包含模型工厂和所需参数的字典。
    """

    def model_factory(**kwargs) -> object:
        """
        模型工厂，用于创建 Darts 模型适配器对象。

        :param kwargs: 模型初始化参数。
        :return: Darts 模型适配器对象。
        """
        return MizarModelAdapter(
            model_name,
            model_class,
            kwargs,
            allow_fit_on_eval,
        )

    return {"model_factory": model_factory, "required_hyper_params": required_args}


MIZAR_MODELS = [
    (ARIMA, {}),
    (InterQuartileRangeAD, {}),
    (StreamSigmaAD, {}),
]

# 针对 MIZAR_MODELS 中的每个模型类和所需参数生成模型工厂并添加到全局变量中
for model_class, required_args in MIZAR_MODELS:
    globals()[f"mizar_{model_class.__name__.lower()}"] = generate_model_factory(
        model_class.__name__, model_class, required_args, allow_fit_on_eval=False
    )
