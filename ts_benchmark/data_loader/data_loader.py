# -*- coding: utf-8 -*-
from functools import reduce
from operator import and_
from typing import List

import pandas as pd

from ts_benchmark.common.constant import META_DETECTION_DATA_PATH
from ts_benchmark.common.constant import META_FORECAST_DATA_PATH

SIZE = {
    "large_forecast": ["large", "medium", "small"],
    "medium_forecast": ["medium", "small"],
    "small_forecast": ["small"],
    "large_detect": ["large", "medium", "small"],
    "medium_detect": ["medium", "small"],
    "small_detect": ["small"],
}


def load_data(data_loader_config: dict) -> List[str]:
    """
    加载数据文件名列表，根据配置筛选文件名。

    :param data_loader_config: 数据加载的配置。
    :return: 符合筛选条件的数据文件名列表。
    :raises RuntimeError: 如果 feature_dict 为 None。
    """
    feature_dict = data_loader_config.get("feature_dict", None)
    if feature_dict is None:
        raise RuntimeError("feature_dict is None")

    # 移除 feature_dict 中值为 None 的项
    feature_dict = {k: v for k, v in feature_dict.items() if v is not None}
    data_set_name = data_loader_config.get("data_set_name", "small_forecast")

    if data_set_name in [
        "large_forecast",
        "medium_forecast",
        "small_forecast",
    ]:
        META_DATA_PATH = META_FORECAST_DATA_PATH
    elif data_set_name in [
        "large_detect",
        "medium_detect",
        "small_detect",
    ]:
        META_DATA_PATH = META_DETECTION_DATA_PATH
    else:
        raise ValueError("请输入正确的data_set_name")

    data_meta = pd.read_csv(META_DATA_PATH)

    data_size = SIZE[data_set_name]
    # 使用 reduce 和 and_ 函数来筛选符合条件的数据文件名
    data_name_list = (
        data_meta[reduce(and_, (data_meta[k] == v for k, v in feature_dict.items()))][
            data_meta["size"].isin(data_size)
        ]
        .iloc[:, 0]
        .tolist()
    )
    return data_name_list
