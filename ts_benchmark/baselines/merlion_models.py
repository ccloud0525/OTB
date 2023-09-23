import numpy as np
import pandas as pd
from merlion.utils import TimeSeries
from merlion.models.anomaly.isolation_forest import (
    IsolationForest,
    IsolationForestConfig,
)
from merlion.models.anomaly.vae import VAE, VAEConfig
from merlion.models.anomaly.windstats import WindStats, WindStatsConfig
from merlion.models.anomaly.autoencoder import AutoEncoder, AutoEncoderConfig
from merlion.models.anomaly.dagmm import DAGMM, DAGMMConfig
from merlion.models.anomaly.dbl import DynamicBaseline, DynamicBaselineConfig
from merlion.models.anomaly.deep_point_anomaly_detector import (
    DeepPointAnomalyDetector,
    DeepPointAnomalyDetectorConfig,
)
from merlion.models.anomaly.lstm_ed import LSTMED, LSTMEDConfig
from merlion.models.anomaly.random_cut_forest import (
    RandomCutForest,
    RandomCutForestConfig,
)
from merlion.models.anomaly.spectral_residual import (
    SpectralResidual,
    SpectralResidualConfig,
)
from merlion.models.anomaly.stat_threshold import StatThreshold, StatThresholdConfig
from merlion.models.anomaly.zms import ZMS, ZMSConfig
from merlion.models.anomaly.change_point.bocpd import BOCPD, BOCPDConfig
from merlion.models.anomaly.forecast_based.arima import (
    ArimaDetector,
    ArimaDetectorConfig,
)
from merlion.models.anomaly.forecast_based.ets import ETSDetector, ETSDetectorConfig
from merlion.models.anomaly.forecast_based.mses import MSESDetector, MSESDetectorConfig


class MerlionModelAdapter:
    """
    Merlion 模型适配器类，用于适配 Merlion 框架中的模型，使其符合预测策略的需求。
    """

    def __init__(
        self,
        model_name: str,
        model_class: object,
        config_class: object,
        model_args: dict,
        allow_label_on_train: bool,
    ):
        """
        初始化 Merlion 模型适配器对象。

        :param model_name: 模型名称。
        :param model_class: Merlion 模型类。
        :param config_class: Merlion 配置类。
        :param model_args: 模型初始化参数。
        :param allow_label_on_train: 是否在训练时使用标签。
        """
        self.model = None
        self.model_class = model_class
        self.config_class = config_class
        self.model_args = model_args
        self.model_name = model_name
        self.allow_label_on_train = allow_label_on_train

    def detect_fit(self, series: pd.DataFrame, label: pd.DataFrame) -> object:
        """
        在时间序列数据上拟合适配的 Merlion 模型。

        :param series: 时间序列数据。
        :param label: 标签数据。
        :return: 拟合后的模型对象。
        """
        self.config_class = self.config_class(**self.model_args)
        self.model = self.model_class(self.config_class)
        series = TimeSeries.from_pd(series)
        label = TimeSeries.from_pd(label)
        if self.allow_label_on_train == True:
            return self.model.train(train_data=series, anomaly_labels=label)
        elif self.allow_label_on_train == False:
            return self.model.train(series)

    def detect_score(self, train: pd.DataFrame) -> np.ndarray:
        """
        使用适配的 Merlion 模型计算异常得分。

        :param train: 用于计算得分的训练数据。
        :return: 异常得分数组。
        """
        train = TimeSeries.from_pd(train)
        fsct_result = self.model.get_anomaly_score(train)

        fsct_result = (fsct_result.to_pd()).reindex((train.to_pd()).index, fill_value=0)

        fsct_result = fsct_result.values.flatten()

        return fsct_result

    def detect_label(self, train: pd.DataFrame) -> np.ndarray:
        """
        使用适配的 Merlion 模型进行异常检测并生成标签。

        :param train: 用于异常检测的训练数据。
        :return: 异常标签数组。
        """
        train = TimeSeries.from_pd(train)
        fsct_result = self.model.get_anomaly_label(train)

        fsct_result = (fsct_result.to_pd()).reindex((train.to_pd()).index, fill_value=0)

        fsct_result = fsct_result.applymap(lambda x: 1 if x != 0 else 0)

        fsct_result = fsct_result.values.flatten()

        return fsct_result

    def __repr__(self):
        """
        返回模型名称的字符串表示。
        """
        return self.model_name


def generate_model_factory(
    model_name: str,
    model_class: object,
    config_class: object,
    required_args: dict,
    allow_label_on_train: bool,
) -> object:
    """
    生成模型工厂信息，用于创建 Merlion 模型适配器。

    :param model_name: 模型名称。
    :param model_class: Merlion 模型类。
    :param config_class: Merlion 配置类。
    :param required_args: 模型初始化所需参数。
    :param allow_label_on_train: 是否在训练时使用标签。
    :return: 包含模型工厂和所需参数的字典。
    """

    def model_factory(**kwargs) -> object:
        """
        模型工厂，用于创建 Merlion 模型适配器对象。

        :param kwargs: 模型初始化参数。
        :return: Merlion 模型适配器对象。
        """
        return MerlionModelAdapter(
            model_name,
            model_class,
            config_class,
            kwargs,
            allow_label_on_train,
        )

    return {"model_factory": model_factory, "required_hyper_params": required_args}


MERLION_MODELS = [
    (IsolationForest, IsolationForestConfig, {}),
    (WindStats, WindStatsConfig, {}),
    (VAE, VAEConfig, {}),
    (AutoEncoder, AutoEncoderConfig, {}),
    (DAGMM, DAGMMConfig, {}),
    (DynamicBaseline, DynamicBaselineConfig, {}),
    (DeepPointAnomalyDetector, DeepPointAnomalyDetectorConfig, {}),
    (LSTMED, LSTMEDConfig, {}),
    (RandomCutForest, RandomCutForestConfig, {}),
    (SpectralResidual, SpectralResidualConfig, {}),
    (StatThreshold, StatThresholdConfig, {}),
    (ZMS, ZMSConfig, {}),
    (BOCPD, BOCPDConfig, {}),
]

MERLION_STAT_MODELS = [  # 训练集不需要标签
    (ArimaDetector, ArimaDetectorConfig, {}),
    (ETSDetector, ETSDetectorConfig, {}),
    (MSESDetector, MSESDetectorConfig, {"max_forecast_steps": "max_forecast_steps"}),
]

# 针对 MERLION_MODELS 中的每个模型类、配置类和所需参数生成模型工厂并添加到全局变量中
for model_class, config_class, required_args in MERLION_MODELS:
    globals()[f"merlion_{model_class.__name__.lower()}"] = generate_model_factory(
        model_class.__name__,
        model_class,
        config_class,
        required_args,
        allow_label_on_train=True,
    )

# 针对 MERLION_STAT_MODELS 中的每个模型类、配置类和所需参数生成模型工厂并添加到全局变量中
for model_class, config_class, required_args in MERLION_STAT_MODELS:
    globals()[f"merlion_{model_class.__name__.lower()}"] = generate_model_factory(
        model_class.__name__,
        model_class,
        config_class,
        required_args,
        allow_label_on_train=False,
    )
