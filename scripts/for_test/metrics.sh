# -*- coding: utf-8 -*-
import warnings

from ts_benchmark.pipeline import pipeline

warnings.filterwarnings("ignore")

# 一 设定data_loader_config：

data_loader_config = {
    "data_set_name": 'small_forecast',
    "feature_dict": {
        "if_univariate": True,
        "if_trend": True,
        "has_timestamp": None,
        "if_season": True,
    },
}

# 二 model_config：

model_config = {
    "models": [
        {
            # "model_name": "lstm.TimeSeriesLSTM",
            # 'adapter': 'darts_deep_model_adapter',
            # 'model_name': 'darts.models.forecasting.arima.ARIMA'
            "model_name": "darts_models.darts_arima",
            "model_hyper_params": {},
        },
        # {
        #     #     # "model_name": "lstm.TimeSeriesLSTM",
        #     #     # 'adapter': 'darts_deep_model_adapter',
        #     "model_name": "darts_models.darts_kalmanforecaster",
        #     #     # "model_hyper_params": {
        #     #     #     "input_chunk_length": 15,
        #     #     #     "output_chunk_length": 24,
        #     #     #     "lstm_layers": 2,
        #     #     #     'n_epochs': 40,
        #     #     # },
        # },
    ],
    "recommend_model_hyper_params": {
        "input_chunk_length": 12,
        "output_chunk_length": 12,
        "add_relative_index": True,
    },
}

# 三 eval_config
model_eval_config = {
    'metric_name': 'all',
    # "metric_name": [
    #     {"name": "mase", "seasonality": 10},
    #     "mae",
    #     {"name": "mase", "seasonality": 2},
    # ],
    # "metric_name": "mae",
    # 'metric_name': {
    #      'name': 'mase',
    #      'seasonality': 10
    #  },
    # 'metric_name': ['mae', 'mse', 'mape', 'smape'],
    "strategy_args": {
        "strategy_name": "fixed_forecast",
        # "strategy_name": "rolling_forecast",
        "pred_len": 12,
        # "train_test_split": 0.8,
        # "stride": 7,
        # "num_rollings": 48,
    },
}

# 四 report_config
report_config = {
    "report_model": "all",
    # 'report_model': ['darts_naivedrift', 'darts_statsforecastautoces'],
    # 'report_model': 'single',
    "report_type": "mean",
    "fill_type": "mean_value",
    "threshold_value": "0.3",
}
pipeline(data_loader_config, model_config, model_eval_config, report_config)

# -*- coding: utf-8 -*-

# from ts_benchmark.pipeline import pipeline
# from merlion.post_process.threshold import AggregateAlarms

# 一 设定data_loader_config：

# data_loader_config = {
#     "feature_dict": {"if_trend": True, "has_timestamp": None, "if_season": True},
# }
#
# # 二 model_config：
#
# model_config = {
#     'models': [
#         # {
#         #     'model_name': 'merlion_models.merlion_autoencoder',
#         # },
#         # {
#         #     'model_name': 'merlion_models.merlion_dagmm',
#         # },
#         # {
#         #     'model_name': 'merlion_models.merlion_dynamicbaseline',
#         # },
#         # {
#         #     'model_name': 'merlion_models.merlion_deeppointanomalydetector',
#         # },
#         # {
#         #     'model_name': 'merlion_models.merlion_lstmed',
#         # },
#         # {
#         #     'model_name': 'merlion_models.merlion_spectralresidual',
#         # },
#         # {
#         #     'model_name': 'merlion_models.merlion_statthreshold',
#         # },
#         # {
#         #     'model_name': 'merlion_models.merlion_zms',
#         # },
#         # {
#         #     'model_name': 'merlion_models.merlion_bocpd',
#         # },
#         # {
#         #     'model_name': 'merlion_models.merlion_arimadetector',
#         # },
#         # {
#         #     'model_name': 'merlion_models.merlion_etsdetector',
#         # },
#         # {
#         #     'model_name': 'lof.LOF',
#         # },
#         {
#             'model_name': 'merlion_models.merlion_isolationforest',
#         },
#         # {
#         #     'model_name': 'merlion_models.merlion_windstats',
#         # },
#         # {
#         #     'model_name': 'merlion_models.merlion_randomcutforest',
#         # },
#         # {
#         #     'model_name': 'merlion_models.merlion_vae',
#         #     "model_hyper_params": {
#         #         'num_epochs': 10,
#         #     },
#         # },
#     ],
#     "recommend_model_hyper_params": {
#         "input_chunk_length": 12,
#         "output_chunk_length": 1,
#         "add_relative_index": True,
#         "max_forecast_steps": 12
#     },
# }
#
# # 三 eval_config
# model_eval_config = {
#     'metric_name': 'all',
#     # 'metric_name': [{
#     #     'name': 'auc_roc',
#     #     'seasonality': 1
#     # }],
#
#     # "metric_name": "mae",
#     # 'metric_name': ['mae', 'mse', 'mape', 'smape'],
#     "strategy_args": {
#         "strategy_name": "fixed_detect_score",
#         # "strategy_name": "rolling_forecast",
#         "pred_len": 15,
#         "train_test_split": 0.1,
#         "stride": 1,
#         "num_rollings": 20,
#     },
# }
#
# # 四 report_config
# report_config = {
#     "report_model": "single",
#     # 'report_model': ['Darts_Croston.csv', 'Darts_TCN.csv'],
#     # 'report_model': 'all',
#     "report_type": "mean",
#     "fill_type": "mean_value",
#     "threshold_value": "0.3",
# }
#
# pipeline(data_loader_config, model_config, model_eval_config, report_config)
