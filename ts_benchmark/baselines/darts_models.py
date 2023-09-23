import logging
from typing import Type, Dict

import darts
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import (
    ARIMA,
    DLinearModel,
    NBEATSModel,
    NLinearModel,
    RNNModel,
    TCNModel,
    AutoARIMA,
    StatsForecastAutoARIMA,
    ExponentialSmoothing,
    StatsForecastAutoETS,
    StatsForecastAutoCES,
    StatsForecastAutoTheta,
    FourTheta,
    FFT,
    KalmanForecaster,
    Croston,
    RegressionModel,
    RandomForest,
    LinearRegressionModel,
    LightGBMModel,
    CatBoostModel,
    XGBModel,
    BlockRNNModel,
    NHiTSModel,
    TransformerModel,
    TFTModel,
    NaiveDrift,
    VARIMA,
)

if darts.__version__ >= "0.25.0":
    from darts.models.utils import NotImportedModule

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class DartsModelAdapter:
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

    def forecast_fit(
        self, series: pd.DataFrame, val_series: pd.DataFrame = pd.DataFrame()
    ) -> object:
        """
        在时间序列数据上拟合适配的 Darts 模型。

        :param series: 时间序列数据。
        :return: 拟合后的模型对象。
        """
        self.model = self.model_class(**self.model_args)
        series = TimeSeries.from_dataframe(series)

        return self.model.fit(series)

    def forecast(self, pred_len: int, train: pd.DataFrame) -> np.ndarray:
        """
        使用适配的 Darts 模型进行预测。

        :param pred_len: 预测长度。
        :param train: 用于拟合模型的训练数据。
        :return: 预测结果。
        """
        if self.allow_fit_on_eval:
            self.forecast_fit(train)
            fsct_result = self.model.predict(pred_len)
        else:
            train = TimeSeries.from_dataframe(train)
            fsct_result = self.model.predict(pred_len, train)
        predict = fsct_result.values()
        return predict

    def __repr__(self):
        """
        返回模型名称的字符串表示。
        """
        return self.model_name


def generate_model_factory(
    model_name: str, model_class: object, required_args: dict, allow_fit_on_eval: bool
) -> Dict:
    """
    生成模型工厂信息，用于创建 Darts 模型适配器。

    :param model_name: 模型名称。
    :param model_class: Darts 模型类。
    :param required_args: 模型初始化所需参数。
    :param allow_fit_on_eval: 是否允许在预测阶段拟合模型。
    :return: 包含模型工厂和所需参数的字典。
    """

    def model_factory(**kwargs) -> DartsModelAdapter:
        """
        模型工厂，用于创建 Darts 模型适配器对象。

        :param kwargs: 模型初始化参数。
        :return: Darts 模型适配器对象。
        """
        return DartsModelAdapter(
            model_name,
            model_class,
            kwargs,
            allow_fit_on_eval,
        )

    return {"model_factory": model_factory, "required_hyper_params": required_args}


DARTS_DEEP_MODEL_DEFAULT_ARGS1 = {
    "input_chunk_length": "input_chunk_length",
    "output_chunk_length": "output_chunk_length",
}
DARTS_DEEP_MODEL_DEFAULT_ARGS2 = {"lags": "input_chunk_length"}

DARTS_MODELS = [
    (ARIMA, {}),
    (VARIMA, {}),
    (KalmanForecaster, {}),
    (TCNModel, DARTS_DEEP_MODEL_DEFAULT_ARGS1),
    (
        TFTModel,
        {
            "input_chunk_length": "input_chunk_length",
            "output_chunk_length": "output_chunk_length",
            "add_relative_index": "add_relative_index",
        },
    ),
    (TransformerModel, DARTS_DEEP_MODEL_DEFAULT_ARGS1),
    (NHiTSModel, DARTS_DEEP_MODEL_DEFAULT_ARGS1),
    (BlockRNNModel, DARTS_DEEP_MODEL_DEFAULT_ARGS1),
    (RNNModel, DARTS_DEEP_MODEL_DEFAULT_ARGS1),
    (DLinearModel, DARTS_DEEP_MODEL_DEFAULT_ARGS1),
    (NBEATSModel, DARTS_DEEP_MODEL_DEFAULT_ARGS1),
    (NLinearModel, DARTS_DEEP_MODEL_DEFAULT_ARGS1),
    (RandomForest, DARTS_DEEP_MODEL_DEFAULT_ARGS2),
    (XGBModel, DARTS_DEEP_MODEL_DEFAULT_ARGS2),
    (CatBoostModel, DARTS_DEEP_MODEL_DEFAULT_ARGS2),
    (LightGBMModel, DARTS_DEEP_MODEL_DEFAULT_ARGS2),
    (LinearRegressionModel, DARTS_DEEP_MODEL_DEFAULT_ARGS2),
    (RegressionModel, DARTS_DEEP_MODEL_DEFAULT_ARGS2),
]

DARTS_STAT_MODELS = [  # 特别允许推理时重新训练
    (AutoARIMA, {}),
    (StatsForecastAutoCES, {}),
    (StatsForecastAutoTheta, {}),
    (StatsForecastAutoETS, {}),
    (ExponentialSmoothing, {}),
    (StatsForecastAutoARIMA, {}),
    (FFT, {}),
    (FourTheta, {}),
    (Croston, {}),
    (NaiveDrift, {}),
]

# 针对 DARTS_MODELS 中的每个模型类和所需参数生成模型工厂并添加到全局变量中
for model_class, required_args in DARTS_MODELS:
    if darts.__version__ >= "0.25.0" and isinstance(model_class, NotImportedModule):
        logger.warning("NotImportedModule encountered, skipping")
        continue
    globals()[f"darts_{model_class.__name__.lower()}"] = generate_model_factory(
        model_class.__name__, model_class, required_args, allow_fit_on_eval=False
    )

# 针对 DARTS_STAT_MODELS 中的每个模型类和所需参数生成模型工厂并添加到全局变量中
for model_class, required_args in DARTS_STAT_MODELS:
    if darts.__version__ >= "0.25.0" and isinstance(model_class, NotImportedModule):
        logger.warning("NotImportedModule encountered, skipping")
        continue
    globals()[f"darts_{model_class.__name__.lower()}"] = generate_model_factory(
        model_class.__name__, model_class, required_args, allow_fit_on_eval=True
    )


def deep_darts_model_adapter(model_info: Type[object]) -> object:
    """
    适配深度 DARTS 模型。

    :param model_info: 要适配的深度 DARTS 模型类。必须是一个类或类型对象。
    :return: 生成的模型工厂，用于创建适配的 DARTS 模型。
    """
    if not isinstance(model_info, type):
        raise ValueError()

    return generate_model_factory(
        model_info.__name__,
        model_info,
        DARTS_DEEP_MODEL_DEFAULT_ARGS1,
        allow_fit_on_eval=False,
    )


def statistics_darts_model_adapter(model_info: Type[object]) -> object:
    """
    适配统计学 DARTS 模型。

    :param model_info: 要适配的统计学 DARTS 模型类。必须是一个类或类型对象。
    :return: 生成的模型工厂，用于创建适配的 DARTS 模型。
    """
    if not isinstance(model_info, type):
        raise ValueError()

    return generate_model_factory(
        model_info.__name__, model_info, {}, allow_fit_on_eval=True
    )