# -*- coding: utf-8 -*-
import os

# 获取代码文件所在的根路径
ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

# # 构建元数据文件的路径
META_FORECAST_DATA_PATH = os.path.join(ROOT_PATH, "dataset", "FORECAST_META.csv")

META_DETECTION_DATA_PATH = os.path.join(ROOT_PATH, "dataset", "DETECT_META.csv")

# 构建数据集文件夹的路径
DATASET_PATH = os.path.join(ROOT_PATH, "dataset")

# # 构建元数据文件的路径
# META_FORECAST_DATA_PATH = os.path.join(ROOT_PATH, "dataset", "formal_forecast", "FORECAST_META.csv")
#
# META_DETECTION_DATA_PATH = os.path.join(ROOT_PATH, "dataset", "ad", "DETECT_META.csv")
# #
# # 构建数据集文件夹的路径
# # DATASET_PATH = os.path.join(ROOT_PATH, "dataset", "ad", "ad_univariate_datasets")
# DATASET_PATH = os.path.join(ROOT_PATH, "dataset", "formal_forecast", "8000")

# 配置文件路径
CONFIG_PATH = os.path.join(ROOT_PATH, "ts_benchmark", "config")

