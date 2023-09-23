import logging
from typing import Type

import numpy as np
import pandas as pd
import os

from ts_benchmark.baselines.third_party.tods.sk_interface.detection_algorithm.IsolationForest_skinterface import IsolationForestSKI
from ts_benchmark.baselines.third_party.tods.sk_interface.detection_algorithm.LSTMODetector_skinterface import LSTMODetectorSKI
from ts_benchmark.baselines.third_party.tods.sk_interface.detection_algorithm.KNN_skinterface import KNNSKI
from ts_benchmark.baselines.third_party.tods.sk_interface.detection_algorithm.AutoEncoder_skinterface import AutoEncoderSKI
from ts_benchmark.baselines.third_party.tods.sk_interface.detection_algorithm.LOF_skinterface import LOFSKI
from ts_benchmark.baselines.third_party.tods.sk_interface.detection_algorithm.OCSVM_skinterface import OCSVMSKI
from ts_benchmark.baselines.third_party.tods.sk_interface.detection_algorithm.HBOS_skinterface import HBOSSKI
from ts_benchmark.baselines.third_party.tods.sk_interface.detection_algorithm.LODA_skinterface import LODASKI


TODS_MODELS =[
    [IsolationForestSKI, {}],
    [LSTMODetectorSKI, {}],
    [KNNSKI, {}],
    [AutoEncoderSKI, {}],
    [LOFSKI, {}],
    [OCSVMSKI, {}],
    [HBOSSKI, {}],
    [LODASKI, {}],

]


logger = logging.getLogger(__name__)


class TodsModelAdapter:
    """
    Tods 模型适配器类，用于适配 Tods 框架中的模型，使其符合预测策略的需求。
    """

    def __init__(
        self,
        model_name: str,
        model_class: object,
        model_args: dict,
    ):
        """
        初始化 Tods 模型适配器对象。

        :param model_name: 模型名称。
        :param model_class: Tods 模型类。
        :param config_class: Tods 配置类。
        :param model_args: 模型初始化参数。
        :param allow_label_on_train: 是否在训练时使用标签。
        """
        self.model = None
        self.model_class = model_class
        self.model_args = model_args
        self.model_name = model_name

    def detect_fit(self, series: pd.DataFrame, label: pd.DataFrame) -> object:
        """
        在时间序列数据上拟合适配的 Tods 模型。

        :param series: 时间序列数据。
        :param label: 标签数据。
        :return: 拟合后的模型对象。
        """
        self.model = self.model_class(**self.model_args)

        return self.model

    def detect_score(self, train: pd.DataFrame) -> np.ndarray:
        """
        使用适配的 Tods 模型计算异常得分。

        :param train: 用于计算得分的训练数据。
        :return: 异常得分数组。
        """
        X = train.values.reshape(-1, 1)
        self.model.fit(X)
        prediction_score = self.model.predict_score(X).reshape(-1)

        return prediction_score

    def detect_label(self, train: pd.DataFrame) -> np.ndarray:
        """
        使用适配的 Tods 模型进行异常检测并生成标签。

        :param train: 用于异常检测的训练数据。
        :return: 异常标签数组。
        """
        X = train.values.reshape(-1, 1)
        self.model.fit(X)
        prediction_labels = self.model.predict(X).reshape(-1)

        return prediction_labels

    def __repr__(self):
        """
        返回模型名称的字符串表示。
        """
        return self.model_name
    
    

def generate_model_factory(
    model_name: str, model_class: object, required_args: dict, 
) -> object:
    """
    生成模型工厂信息，用于创建 Tods 模型适配器。

    :param model_name: 模型名称。
    :param model_class: Tods 模型类。
    :param required_args: 模型初始化所需参数。
    :return: 包含模型工厂和所需参数的字典。
    """

    def model_factory(**kwargs) -> object:
        """
        模型工厂，用于创建 Tods 模型适配器对象。
        :param kwargs: 模型初始化参数。
        :return: Tods 模型适配器对象。
        """
        return TodsModelAdapter(model_name, model_class, kwargs,)

    return {"model_factory": model_factory, "required_hyper_params": required_args}




# 针对 TODS_MODELS 中的每个模型类和所需参数生成模型工厂并添加到全局变量中
for model_class, required_args in TODS_MODELS:
    globals()[f"tods_{model_class.__name__.lower()}"] = generate_model_factory(
        model_class.__name__, model_class, required_args
    )


#TODO tods adapter
# def deep_tods_model_adapter(model_info: Type[object]) -> object:
#     """
#     适配深度 Tods 模型。

#     :param model_info: 要适配的深度 Tods 模型类。必须是一个类或类型对象。
#     :return: 生成的模型工厂，用于创建适配的 Tods 模型。
#     """
#     if not isinstance(model_info, type):
#         raise ValueError()

#     return generate_model_factory(
#         model_info.__name__,
#         model_info,
#         allow_fit_on_eval=False,
#     )


