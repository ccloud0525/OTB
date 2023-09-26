# -*- coding: utf-8 -*-
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Tuple

import pandas as pd

from ts_benchmark.common.constant import DATASET_PATH
from ts_benchmark.common.constant import META_DETECTION_DATA_PATH
from ts_benchmark.common.constant import META_FORECAST_DATA_PATH
from ts_benchmark.utils.data_processing import read_data


class Singleton(type):
    """
    用于通过meta class的方法构造单例类
    """

    _instance_dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance_dict:
            cls._instance_dict[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance_dict[cls]


class DataPool(metaclass=Singleton):
    """
    DataPool 类用于创建数据池，加速读取多个数据文件。
    """

    _DATA_KEY = "file_name"

    def __init__(self):
        """
        构造函数，初始化 DataPool 实例。
        """
        self.forecast_data_meta = pd.read_csv(META_FORECAST_DATA_PATH)
        self.detect_data_meta = pd.read_csv(META_DETECTION_DATA_PATH)
        self.forecast_data_meta.set_index(self._DATA_KEY, drop=False, inplace=True)
        self.detect_data_meta.set_index(self._DATA_KEY, drop=False, inplace=True)

        self.data_pool = {}  # 创建一个字典用于存储数据

    def _load_meta_info(self, series_name: str) -> Union[pd.Series, None]:
        """
        准备指定系列的元数据信息。

        :param series_name: 要查找元数据的系列名称。
        :return: 包含元数据信息的 pandas Series。
        :raises ValueError: 如果没有找到指定系列名称的元数据信息。
        """
        if series_name in self.forecast_data_meta.index:
            return self.forecast_data_meta.loc[[series_name]]
        elif series_name in self.detect_data_meta.index:
            return self.detect_data_meta.loc[[series_name]]
        else:
            raise ValueError("do not have {}'s meta data".format(series_name))

    def prepare_data(self, list_of_files: list) -> None:
        """
        并行加载多个数据文件到数据池。

        :param list_of_files: 要加载的数据文件列表。
        """
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._load_data, series_name)
                for series_name in list_of_files
            ]
            for future, series_name in zip(futures, list_of_files):
                self.data_pool[series_name] = future.result()

    def _load_data(self, series_name: str) -> tuple:
        """
        加载单个数据文件并返回文件名和数据。

        :param series_name: 数据文件名。
        :return: 包含文件名和元数据的二元组。
        """
        datafile_path = os.path.join(DATASET_PATH, series_name)
        data = read_data(datafile_path)
        return data, self._load_meta_info(series_name)

    def get_series(self, series_name: str) -> pd.DataFrame:
        """
        根据文件名获取数据池中的数据。

        :param series_name: 数据文件名。
        :return: 对应的数据。
        :raises ValueError: 如果数据文件不在数据池中。
        """
        if series_name not in self.data_pool:
            self.data_pool[series_name] = self._load_data(series_name)
        return self.data_pool[series_name][0]

    def get_series_meta_info(self, series_name: str) -> pd.DataFrame:
        """
        根据文件名获取数据池中的数据。

        :param series_name: 数据文件名。
        :return: 对应的数据meta信息。
        :raises ValueError: 如果数据文件不在数据池中。
        """
        if series_name not in self.data_pool:
            self.data_pool[series_name] = self._load_data(series_name)
        return self.data_pool[series_name][1]
