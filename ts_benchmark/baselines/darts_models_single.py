# import logging
# import os
# from typing import Dict
#
# import darts
# import numpy as np
# import pandas as pd
# from darts import TimeSeries
# from darts.models import (
#     ARIMA,
#     DLinearModel,
#     NBEATSModel,
#     NLinearModel,
#     RNNModel,
#     TCNModel,
#     AutoARIMA,
#     StatsForecastAutoARIMA,
#     ExponentialSmoothing,
#     StatsForecastAutoETS,
#     StatsForecastAutoCES,
#     StatsForecastAutoTheta,
#     FourTheta,
#     FFT,
#     KalmanForecaster,
#     Croston,
#     RegressionModel,
#     RandomForest,
#     LinearRegressionModel,
#     LightGBMModel,
#     CatBoostModel,
#     XGBModel,
#     BlockRNNModel,
#     NHiTSModel,
#     TransformerModel,
#     TFTModel,
#     TiDEModel,
#     NaiveDrift,
#     VARIMA,
# )
# from sklearn.preprocessing import StandardScaler
#
# from ts_benchmark.baselines.utils import train_val_split
# from ts_benchmark.utils.data_processing import split_before
#
# if darts.__version__ >= "0.25.0":
#     from darts.models.utils import NotImportedModule
#
# logger = logging.getLogger(__name__)
#
#
# class DartsModelAdapter:
#     """
#     Darts 模型适配器类，用于适配 Darts 框架中的模型，使其符合预测策略的需求。
#     """
#
#     def __init__(
#         self,
#         model_name: str,
#         model_class: object,
#         model_args: dict,
#         allow_fit_on_eval: bool,
#     ):
#         """
#         初始化 Darts 模型适配器对象。
#
#         :param model_name: 模型名称。
#         :param model_class: Darts 模型类。
#         :param model_args: 模型初始化参数。
#         :param allow_fit_on_eval: 是否允许在预测阶段拟合模型。
#         """
#         self.model = None
#         self.model_class = model_class
#         self.model_args = model_args
#         self.model_name = model_name
#         self.allow_fit_on_eval = allow_fit_on_eval
#         self.scaler = StandardScaler()
#
#
#     # def forecast_fit(
#     #     self, series: pd.DataFrame, val_series: pd.DataFrame = pd.DataFrame()
#     # ) -> object:
#     #     """
#     #     在时间序列数据上拟合适配的 Darts 模型。
#     #
#     #     :param series: 时间序列数据。
#     #     :return: 拟合后的模型对象。
#     #     """
#     #     # TODO: training and inferencing on multiple gpus with 'ddp' strategy is error prone
#     #     #  in complicated work flow, the problems include but not limited to:
#     #     #  - do heavy initialization in all processes (e.g. full data loading)
#     #     #  - hangs when the program is interrupted (e.g. exceptions that are caught elsewhere)
#     #     #  - not compatible with the parallel paradigm of ray
#     #     #  As a result, we disallow a single worker to work on multiple gpus by now, but what if
#     #     #  evaluating large-scale models is required in the future?
#     #     gpu_devices = list(
#     #         filter(None, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
#     #     )
#     #     if gpu_devices:
#     #         pl_args = self.model_args.get("pl_trainer_kwargs", {})
#     #         device_args = pl_args.get("devices", None)
#     #         if (
#     #             device_args is None
#     #             or (isinstance(device_args, list) and len(device_args) > 1)
#     #             or (isinstance(device_args, int) and device_args > 1)
#     #         ):
#     #             self.model_args.setdefault("pl_trainer_kwargs", {})
#     #             self.model_args["pl_trainer_kwargs"]["devices"] = [int(gpu_devices[0])]
#     #             logger.warning(
#     #                 "Multi-gpu training is not supported, using only gpu %s",
#     #                 self.model_args["pl_trainer_kwargs"]["devices"],
#     #             )
#     #
#     #     self.model = self.model_class(**self.model_args)
#     #     series = TimeSeries.from_dataframe(series)
#     #     border = int((series.shape[0]) * 0.75)
#     #
#     #     train_data, valid_data = split_before(series, border)
#     #     self.model = self.model_class(**self.model_args)
#     #     train_data = TimeSeries.from_dataframe(train_data)
#     #     valid_data = TimeSeries.from_dataframe(valid_data)
#     #
#     #     return self.model.fit(train_data, val_series = valid_data)
#
#
#
#         # self.model = self.model_class(**self.model_args)
#         # series = TimeSeries.from_dataframe(series)
#         #
#         # return self.model.fit(series)
#
#     def forecast_fit(
#             self, series: pd.DataFrame, ratio
#     ) -> object:
#         """
#         在时间序列数据上拟合适配的 Darts 模型。
#
#         :param series: 时间序列数据。
#         :return: 拟合后的模型对象。
#         """
#         # TODO: training and inferencing on multiple gpus with 'ddp' strategy is error prone
#         #  in complicated work flow, the problems include but not limited to:
#         #  - do heavy initialization in all processes (e.g. full data loading)
#         #  - hangs when the program is interrupted (e.g. exceptions that are caught
#         #  elsewhere)
#         #  - not compatible with the parallel paradigm of ray
#         #  As a result, we disallow a single worker to work on multiple gpus by now, but what if
#         #  evaluating large-scale models is required in the future?
#         # gpu_devices = list(
#         #     filter(None, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
#         # )
#         # if gpu_devices:
#         #     pl_args = self.model_args.get("pl_trainer_kwargs", {})
#         #     device_args = pl_args.get("devices", None)
#         #     if (
#         #             device_args is None
#         #             or (isinstance(device_args, list) and len(device_args) > 1)
#         #             or (isinstance(device_args, int) and device_args > 1)
#         #     ):
#         #         self.model_args.setdefault("pl_trainer_kwargs", {})
#         #         # self.model_args["pl_trainer_kwargs"]["devices"] = [int(gpu_devices[0])]
#         #         self.model_args["pl_trainer_kwargs"]["devices"] = [0]
#         #         logger.warning(
#         #             "Multi-gpu training is not supported, using only gpu %s",
#         #             self.model_args["pl_trainer_kwargs"]["devices"],
#         #         )
#
#
#         train_data_value, valid_data = train_val_split(series, ratio, self.model_args.get('input_chunk_length', 0))
#         self.scaler.fit(train_data_value.values)
#         train_data_value = pd.DataFrame(self.scaler.transform(train_data_value.values),
#                                         columns=train_data_value.columns,
#                                         index=train_data_value.index)
#         valid_data = pd.DataFrame(self.scaler.transform(valid_data.values), columns=valid_data.columns,
#                                   index=valid_data.index)
#
#         self.model = self.model_class(**self.model_args)
#         train_data = TimeSeries.from_dataframe(train_data_value)
#         valid_data = TimeSeries.from_dataframe(valid_data)
#
#         return self.model.fit(train_data, val_series=valid_data)
#
#     # def forecast(self, pred_len: int, train: pd.DataFrame) -> np.ndarray:
#     #     """
#     #     使用适配的 Darts 模型进行预测。
#     #
#     #     :param pred_len: 预测长度。
#     #     :param train: 用于拟合模型的训练数据。
#     #     :return: 预测结果。
#     #     """
#     #     if self.allow_fit_on_eval:
#     #         self.forecast_fit(train)
#     #         fsct_result = self.model.predict(pred_len)
#     #     else:
#     #         train = TimeSeries.from_dataframe(train)
#     #         fsct_result = self.model.predict(pred_len, train)
#     #     predict = fsct_result.values()
#     #     return predict
#     def forecast(self, pred_len: int, train: pd.DataFrame) -> np.ndarray:
#         """
#         使用适配的 Darts 模型进行预测。
#
#         :param pred_len: 预测长度。
#         :param train: 用于拟合模型的训练数据。
#         :return: 预测结果。
#         """
#         train = pd.DataFrame(self.scaler.transform(train.values), columns=train.columns,
#                                               index=train.index)
#         if self.allow_fit_on_eval:
#             self.forecast_fit(train)
#             fsct_result = self.model.predict(pred_len)
#         else:
#             train = TimeSeries.from_dataframe(train)
#             fsct_result = self.model.predict(pred_len, train)
#         predict = fsct_result.values()
#         predict = self.scaler.inverse_transform(predict)
#
#         return predict
#
#     def __repr__(self):
#         """
#         返回模型名称的字符串表示。
#         """
#         return self.model_name
#
#
# def generate_model_factory(
#     model_name: str,
#     model_class: object,
#     model_args: dict,
#     required_args: dict,
#     allow_fit_on_eval: bool,
# ) -> Dict:
#     """
#     生成模型工厂信息，用于创建 Darts 模型适配器。
#
#     :param model_name: 模型名称。
#     :param model_class: Darts 模型类。
#     :param model_args: 预定义的模型超参数，可以被输入工厂函数的超参数覆盖。
#     :param required_args: 需要由 benchmark 推荐的超参数。
#     :param allow_fit_on_eval: 是否允许在预测阶段拟合模型。
#     :return: 包含模型工厂和所需参数的字典。
#     """
#
#     def model_factory(**kwargs) -> DartsModelAdapter:
#         """
#         模型工厂，用于创建 Darts 模型适配器对象。
#
#         :param kwargs: 模型初始化参数。
#         :return: Darts 模型适配器对象。
#         """
#         return DartsModelAdapter(
#             model_name,
#             model_class,
#             {**model_args, **kwargs},
#             allow_fit_on_eval,
#         )
#
#     return {"model_factory": model_factory, "required_hyper_params": required_args}
#
#
# DARTS_DEEP_MODEL_REQUIRED_ARGS1 = {
#     "input_chunk_length": "input_chunk_length",
#     "output_chunk_length": "output_chunk_length",
# }
# DARTS_DEEP_MODEL_REQUIRED_ARGS2 = {"lags": "input_chunk_length"}
# DARTS_DEEP_MODEL_ARGS = {
#     "pl_trainer_kwargs": {
#         "enable_progress_bar": False,
#     }
# }
#
# DARTS_MODELS = [
#     (RandomForest, DARTS_DEEP_MODEL_REQUIRED_ARGS2, DARTS_DEEP_MODEL_ARGS),
#     (XGBModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, DARTS_DEEP_MODEL_ARGS),
#     (CatBoostModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, DARTS_DEEP_MODEL_ARGS),
#     (LightGBMModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, DARTS_DEEP_MODEL_ARGS),
#     (LinearRegressionModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, DARTS_DEEP_MODEL_ARGS),
#     (RegressionModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, DARTS_DEEP_MODEL_ARGS),
#     (ARIMA, {}, {}),
#     (VARIMA, {}, {}),
#     (KalmanForecaster, {}, {}),
#     (TCNModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, {}),
#     (
#         TFTModel,
#         DARTS_DEEP_MODEL_REQUIRED_ARGS1,
#         DARTS_DEEP_MODEL_ARGS,
#     ),
#     (TransformerModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
#     (NHiTSModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
#     (TiDEModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
#     (BlockRNNModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
#     (RNNModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
#     (DLinearModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
#     (NBEATSModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
#     (NLinearModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
# ]
#
# # 以下模型特别允许推理时重新训练
# DARTS_STAT_MODELS = [
#     (VARIMA, {}, {}),
#     (AutoARIMA, {}, {}),
#     (CatBoostModel, {}, {}),
#     (LightGBMModel, {}, {}),
#     (StatsForecastAutoCES, {}, {}),
#     (StatsForecastAutoTheta, {}, {}),
#     (StatsForecastAutoETS, {}, {}),
#     (ExponentialSmoothing, {}, {}),
#     (StatsForecastAutoARIMA, {}, {}),
#     (FFT, {}, {}),
#     (FourTheta, {}, {}),
#     (Croston, {}, {}),
#     (NaiveDrift, {}, {}),
# ]
#
# # 针对 DARTS_MODELS 中的每个模型类和所需参数生成模型工厂并添加到全局变量中
# for model_class, required_args, model_args in DARTS_MODELS:
#     if darts.__version__ >= "0.25.0" and isinstance(model_class, NotImportedModule):
#         logger.warning("NotImportedModule encountered, skipping")
#         continue
#     globals()[f"darts_{model_class.__name__.lower()}"] = generate_model_factory(
#         model_class.__name__,
#         model_class,
#         model_args,
#         required_args,
#         allow_fit_on_eval=False,
#     )
#
# # 针对 DARTS_STAT_MODELS 中的每个模型类和所需参数生成模型工厂并添加到全局变量中
# for model_class, required_args, model_args in DARTS_STAT_MODELS:
#     if darts.__version__ >= "0.25.0" and isinstance(model_class, NotImportedModule):
#         logger.warning("NotImportedModule encountered, skipping")
#         continue
#     globals()[f"darts_{model_class.__name__.lower()}"] = generate_model_factory(
#         model_class.__name__,
#         model_class,
#         model_args,
#         required_args,
#         allow_fit_on_eval=True,
#     )
#
#
# # TODO：darts 应该不止这两个 adapter，例如有些应该输入 DARTS_DEEP_MODEL_REQUIRED_ARGS2
# #   而非 DARTS_DEEP_MODEL_REQUIRED_ARGS1。
# #   因此暂时注释这两个 adapter，后续看是去掉这些 adapter 还是通过 inspect 来分析模型参数
# #   还是预先定义好模型与 adapter 之间的映射关系。
# # def deep_darts_model_adapter(model_info: Type[object]) -> object:
# #     """
# #     适配深度 DARTS 模型。
# #
# #     :param model_info: 要适配的深度 DARTS 模型类。必须是一个类或类型对象。
# #     :return: 生成的模型工厂，用于创建适配的 DARTS 模型。
# #     """
# #     if not isinstance(model_info, type):
# #         raise ValueError()
# #
# #     return generate_model_factory(
# #         model_info.__name__,
# #         model_info,
# #         DARTS_DEEP_MODEL_REQUIRED_ARGS1,
# #         allow_fit_on_eval=False,
# #     )
# #
# #
# # def statistics_darts_model_adapter(model_info: Type[object]) -> object:
# #     """
# #     适配统计学 DARTS 模型。
# #
# #     :param model_info: 要适配的统计学 DARTS 模型类。必须是一个类或类型对象。
# #     :return: 生成的模型工厂，用于创建适配的 DARTS 模型。
# #     """
# #     if not isinstance(model_info, type):
# #         raise ValueError()
# #
# #     return generate_model_factory(
# #         model_info.__name__, model_info, {}, allow_fit_on_eval=True
# #     )
import logging
from typing import Dict

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
    TiDEModel,
    NaiveDrift,
    VARIMA,
    NaiveMean,
    NaiveSeasonal,
    NaiveMovingAverage,
)
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.utils.data_processing import split_before

if darts.__version__ >= "0.25.0":
    from darts.models.utils import NotImportedModule

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
        self.scaler = StandardScaler()

    def forecast_fit(self, series: pd.DataFrame, ratio) -> object:
        """
        在时间序列数据上拟合适配的 Darts 模型。

        :param series: 时间序列数据。
        :return: 拟合后的模型对象。
        """
        # TODO: training and inferencing on multiple gpus with 'ddp' strategy is error prone
        #  in complicated work flow, the problems include but not limited to:
        #  - do heavy initialization in all processes (e.g. full data loading)
        #  - hangs when the program is interrupted (e.g. exceptions that are caught
        #  elsewhere)
        #  - not compatible with the parallel paradigm of ray
        #  As a result, we disallow a single worker to work on multiple gpus by now, but what if
        #  evaluating large-scale models is required in the future?
        # gpu_devices = list(
        #     filter(None, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
        # )
        # if gpu_devices:
        #     pl_args = self.model_args.get("pl_trainer_kwargs", {})
        #     device_args = pl_args.get("devices", None)
        #     if (
        #             device_args is None
        #             or (isinstance(device_args, list) and len(device_args) > 1)
        #             or (isinstance(device_args, int) and device_args > 1)
        #     ):
        #         self.model_args.setdefault("pl_trainer_kwargs", {})
        #         # self.model_args["pl_trainer_kwargs"]["devices"] = [int(gpu_devices[0])]
        #         self.model_args["pl_trainer_kwargs"]["devices"] = [0]
        #         logger.warning(
        #             "Multi-gpu training is not supported, using only gpu %s",
        #             self.model_args["pl_trainer_kwargs"]["devices"],
        #         )
        print(
            "----------------------------------------------------------",
            self.model_name,
        )

        if self.model_name == "VARIMA":
            self.model = self.model_class(**self.model_args)
            # self.scaler.fit(series.values)
            # train_data_value = pd.DataFrame(self.scaler.transform(series.values),
            #                                 columns=series.columns,
            #                                 index=series.index)
            train_data = TimeSeries.from_dataframe(series)
            # train_data = series
            return self.model.fit(train_data)
        else:
            # self.scaler.fit(series.values)
            # train_data_value = pd.DataFrame(self.scaler.transform(series.values),
            #                                 columns=series.columns,
            #                                 index=series.index)

            train_data_value = series
            self.model = self.model_class(**self.model_args)
            train_data = TimeSeries.from_dataframe(train_data_value)

            return self.model.fit(train_data)

    def forecast(self, pred_len: int, train: pd.DataFrame) -> np.ndarray:
        """
        使用适配的 Darts 模型进行预测。

        :param pred_len: 预测长度。
        :param train: 用于拟合模型的训练数据。
        :return: 预测结果。
        """
        # train = pd.DataFrame(self.scaler.transform(train.values), columns=train.columns,
        #                                       index=train.index)
        if self.allow_fit_on_eval:
            fsct_result = self.model.predict(pred_len)
        else:
            train = TimeSeries.from_dataframe(train)
            fsct_result = self.model.predict(pred_len, train)
        predict = fsct_result.values()
        # predict = self.scaler.inverse_transform(predict)

        return predict

    def inner_forecast_back(
        self, horizon_len: int, pred_len: int, data: pd.DataFrame
    ) -> np.ndarray:

        output = []
        for i in range(0, data.shape[0] - pred_len - horizon_len + 1):
            x=data.iloc[0: i + horizon_len]
            train = TimeSeries.from_dataframe(x)
            fsct_result = self.model.predict(pred_len, train).values()
            output.append(fsct_result)

        output = np.array(output, dtype=float)
        return output

    # def forecast_fit(
    #     self, series: pd.DataFrame, val_series: pd.DataFrame = pd.DataFrame()
    # ) -> object:
    #     """
    #     在时间序列数据上拟合适配的 Darts 模型。
    #
    #     :param series: 时间序列数据。
    #     :return: 拟合后的模型对象。
    #     """
    #     border = int((series.shape[0]) * 0.75)
    #
    #     train_data, valid_data = split_before(series, border)
    #     self.model = self.model_class(**self.model_args)
    #     train_data = TimeSeries.from_dataframe(train_data)
    #     valid_data = TimeSeries.from_dataframe(valid_data)
    #
    #     return self.model.fit(train_data, val_series = valid_data)
    #
    #     # self.model = self.model_class(**self.model_args)
    #     # series = TimeSeries.from_dataframe(series)
    #     #
    #     # return self.model.fit(series)
    #
    # def forecast(self, pred_len: int, train: pd.DataFrame) -> np.ndarray:
    #     """
    #     使用适配的 Darts 模型进行预测。
    #
    #     :param pred_len: 预测长度。
    #     :param train: 用于拟合模型的训练数据。
    #     :return: 预测结果。
    #     """
    #     if self.allow_fit_on_eval:
    #         self.forecast_fit(train)
    #         fsct_result = self.model.predict(pred_len)
    #     else:
    #         train = TimeSeries.from_dataframe(train)
    #         fsct_result = self.model.predict(pred_len, train)
    #     predict = fsct_result.values()
    #     return predict

    def __repr__(self):
        """
        返回模型名称的字符串表示。
        """
        return self.model_name


def generate_model_factory(
    model_name: str,
    model_class: object,
    model_args: dict,
    required_args: dict,
    allow_fit_on_eval: bool,
) -> Dict:
    """
    生成模型工厂信息，用于创建 Darts 模型适配器。

    :param model_name: 模型名称。
    :param model_class: Darts 模型类。
    :param model_args: 预定义的模型超参数，可以被输入工厂函数的超参数覆盖。
    :param required_args: 需要由 benchmark 推荐的超参数。
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
            {**model_args, **kwargs},
            allow_fit_on_eval,
        )

    return {"model_factory": model_factory, "required_hyper_params": required_args}


DARTS_DEEP_MODEL_REQUIRED_ARGS1 = {
    "input_chunk_length": "input_chunk_length",
    "output_chunk_length": "output_chunk_length",
}
DARTS_DEEP_MODEL_REQUIRED_ARGS2 = {"lags": "input_chunk_length"}
DARTS_DEEP_MODEL_REQUIRED_ARGS3 = {
    "lags": "input_chunk_length",
    "output_chunk_length": "output_chunk_length",
}
DARTS_DEEP_MODEL_ARGS = {
    "pl_trainer_kwargs": {
        "enable_progress_bar": False,
    }
}

DARTS_MODELS = [
    (KalmanForecaster, {}, {}),
    (TCNModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (
        TFTModel,
        DARTS_DEEP_MODEL_REQUIRED_ARGS1,
        {},
    ),
    (TransformerModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (NHiTSModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (TiDEModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (BlockRNNModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (RNNModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (DLinearModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (NBEATSModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (NLinearModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    # (RandomForest, DARTS_DEEP_MODEL_REQUIRED_ARGS2, DARTS_DEEP_MODEL_ARGS),
    (RandomForest, DARTS_DEEP_MODEL_REQUIRED_ARGS2, {}),
    (XGBModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, DARTS_DEEP_MODEL_ARGS),
    (CatBoostModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, {}),
    # (CatBoostModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, DARTS_DEEP_MODEL_ARGS),
    # (LightGBMModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, DARTS_DEEP_MODEL_ARGS),
    (LightGBMModel, DARTS_DEEP_MODEL_REQUIRED_ARGS3, {}),
    (LinearRegressionModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, {}),
    (RegressionModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, {}),
    # (RegressionModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, DARTS_DEEP_MODEL_ARGS),
]

# 以下模型特别允许推理时重新训练
DARTS_STAT_MODELS = [
    (ARIMA, {}, {}),
    (VARIMA, {}, {}),
    (AutoARIMA, {}, {}),
    (StatsForecastAutoCES, {}, {}),
    (StatsForecastAutoTheta, {}, {}),
    (StatsForecastAutoETS, {}, {}),
    (ExponentialSmoothing, {}, {}),
    (StatsForecastAutoARIMA, {}, {}),
    (FFT, {}, {}),
    (FourTheta, {}, {}),
    (Croston, {}, {}),
    (NaiveDrift, {}, {}),
    (NaiveMean, {}, {}),
    (NaiveSeasonal, {}, {}),
    (NaiveMovingAverage, {}, {}),
]

# 针对 DARTS_MODELS 中的每个模型类和所需参数生成模型工厂并添加到全局变量中
for model_class, required_args, model_args in DARTS_MODELS:
    if darts.__version__ >= "0.25.0" and isinstance(model_class, NotImportedModule):
        logger.warning("NotImportedModule encountered, skipping")
        continue
    globals()[f"darts_{model_class.__name__.lower()}"] = generate_model_factory(
        model_class.__name__,
        model_class,
        model_args,
        required_args,
        allow_fit_on_eval=False,
    )

# 针对 DARTS_STAT_MODELS 中的每个模型类和所需参数生成模型工厂并添加到全局变量中
for model_class, required_args, model_args in DARTS_STAT_MODELS:
    if darts.__version__ >= "0.25.0" and isinstance(model_class, NotImportedModule):
        logger.warning("NotImportedModule encountered, skipping")
        continue
    globals()[f"darts_{model_class.__name__.lower()}"] = generate_model_factory(
        model_class.__name__,
        model_class,
        model_args,
        required_args,
        allow_fit_on_eval=True,
    )

# TODO：darts 应该不止这两个 adapter，例如有些应该输入 DARTS_DEEP_MODEL_REQUIRED_ARGS2
#   而非 DARTS_DEEP_MODEL_REQUIRED_ARGS1。
#   因此暂时注释这两个 adapter，后续看是去掉这些 adapter 还是通过 inspect 来分析模型参数
#   还是预先定义好模型与 adapter 之间的映射关系。
# def deep_darts_model_adapter(model_info: Type[object]) -> object:
#     """
#     适配深度 DARTS 模型。
#
#     :param model_info: 要适配的深度 DARTS 模型类。必须是一个类或类型对象。
#     :return: 生成的模型工厂，用于创建适配的 DARTS 模型。
#     """
#     if not isinstance(model_info, type):
#         raise ValueError()
#
#     return generate_model_factory(
#         model_info.__name__,
#         model_info,
#         DARTS_DEEP_MODEL_REQUIRED_ARGS1,
#         allow_fit_on_eval=False,
#     )
#
#
# def statistics_darts_model_adapter(model_info: Type[object]) -> object:
#     """
#     适配统计学 DARTS 模型。
#
#     :param model_info: 要适配的统计学 DARTS 模型类。必须是一个类或类型对象。
#     :return: 生成的模型工厂，用于创建适配的 DARTS 模型。
#     """
#     if not isinstance(model_info, type):
#         raise ValueError()
#
#     return generate_model_factory(
#         model_info.__name__, model_info, {}, allow_fit_on_eval=True
#     )
