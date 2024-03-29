# -*- coding: utf-8 -*-
from typing import Tuple

import pandas as pd
import torch

from ts_benchmark.baselines.time_series_library.utils.timefeatures import time_features


class SlidingWindowDataLoader:
    """
    SlidingWindowDataLoader 类。

    该类封装了滑动窗口数据加载器，用于生成时间序列训练样本。
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        batch_size: int = 1,
        history_length: int = 10,
        prediction_length: int = 2,
        shuffle: bool = True,
    ):
        """
        初始化 SlidingWindowDataLoader。

        :param dataset: 包含时间序列数据的 pandas DataFrame。
        :param batch_size: 批次大小。
        :param history_length: 历史数据的长度。
        :param prediction_length: 预测数据的长度。
        :param shuffle: 是否对数据集进行洗牌。
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.history_length = history_length
        self.prediction_length = prediction_length
        self.shuffle = shuffle
        self.current_index = 0

    def __len__(self) -> int:
        """
        返回数据加载器的长度。

        :return: 数据加载器的长度。
        """
        return len(self.dataset) - self.history_length - self.prediction_length + 1

    def __iter__(self) -> "SlidingWindowDataLoader":
        """
        创建迭代器并返回。

        :return: 数据加载器迭代器。
        """
        if self.shuffle:
            self._shuffle_dataset()
        self.current_index = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成下一个批次的数据。

        :return: 包含输入数据和目标数据的元组。
        """
        if self.current_index >= len(self):
            raise StopIteration

        batch_inputs = []
        batch_targets = []
        for _ in range(self.batch_size):
            window_data = self.dataset.iloc[
                self.current_index : self.current_index
                + self.history_length
                + self.prediction_length,
                :,
            ]
            if len(window_data) < self.history_length + self.prediction_length:
                raise StopIteration  # 数据集不足一个窗口大小和预测步长时，停止迭代

            inputs = window_data.iloc[: self.history_length].values
            targets = window_data.iloc[
                self.history_length : self.history_length + self.prediction_length
            ].values

            batch_inputs.append(inputs)
            batch_targets.append(targets)
            self.current_index += 1

        # 将NumPy数组转换为PyTorch张量
        batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32)
        batch_targets = torch.tensor(batch_targets, dtype=torch.float32)

        return batch_inputs, batch_targets

    def _shuffle_dataset(self):
        """
        对数据集进行洗牌。
        """
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)


class DataloaderForTransformer:
    def __init__(
        self,
        dataset: pd.DataFrame,
        batch_size: int = 1,
        history_len: int = 10,
        prediction_len: int = 2,
        label_len: int = 5,
        shuffle: bool = True,
        timeenc: int = 0,
        freq: str = "h",
        seasonal_patterns=None,
    ):
        # init

        self.dataset = dataset
        self.batch_size = batch_size
        self.history_length = history_len
        self.prediction_length = prediction_len
        self.label_length = label_len
        self.shuffle = shuffle
        self.current_index = 0
        self.timeenc = timeenc
        self.freq = freq

        self.__read_data__()

    def __len__(self) -> int:
        """
        返回数据加载器的长度。

        :return: 数据加载器的长度。
        """
        return len(self.dataset) - self.history_length - self.prediction_length + 1

    def __iter__(self) -> "DataloaderForTransformer":
        """
        创建迭代器并返回。

        :return: 数据加载器迭代器。
        """
        if self.shuffle:
            self._shuffle_dataset()
        self.current_index = 0

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成下一个批次的数据。

        :return: 包含输入数据和目标数据的元组。
        """
        if self.current_index >= len(self):
            raise StopIteration

        batch_inputs = []
        batch_targets = []
        batch_inputs_mark = []
        batch_targets_mark = []
        for _ in range(self.batch_size):
            window_data = self.dataset.iloc[
                self.current_index : self.current_index
                + self.history_length
                + self.prediction_length,
                :,
            ]
            if len(window_data) < self.history_length + self.prediction_length:
                raise StopIteration  # 数据集不足一个窗口大小和预测步长时，停止迭代

            inputs = window_data.iloc[: self.history_length].values
            targets = window_data.iloc[
                self.history_length
                - self.label_length : self.history_length
                + self.prediction_length
            ].values

            batch_inputs.append(inputs)
            batch_targets.append(targets)

            window_mark = self.data_stamp.iloc[
                self.current_index : self.current_index
                + self.history_length
                + self.prediction_length,
            ]
            inputs_mark = window_mark.iloc[: self.history_length].values
            targets_mark = window_mark.iloc[
                self.history_length
                - self.label_length : self.history_length
                + self.prediction_length
            ].values
            batch_inputs_mark.append(inputs_mark)
            batch_targets_mark.append(targets_mark)
            self.current_index += 1

        # 将NumPy数组转换为PyTorch张量
        batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32)
        batch_targets = torch.tensor(batch_targets, dtype=torch.float32)
        batch_inputs_mark = torch.tensor(batch_inputs_mark, dtype=torch.float32)
        batch_targets_mark = torch.tensor(batch_targets_mark, dtype=torch.float32)

        return batch_inputs, batch_targets, batch_inputs_mark, batch_targets_mark

    def _shuffle_dataset(self):
        """
        对数据集进行洗牌。
        """
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)
        self.__read_data__()

    def __read_data__(self):
        df_stamp = self.dataset.reset_index()
        df_stamp = df_stamp[["date"]]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            if self.freq != "m":
                df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
                if self.freq != "w":
                    df_stamp["weekday"] = df_stamp.date.apply(
                        lambda row: row.weekday(), 1
                    )
                    if self.freq != "b" and self.freq != "d":
                        df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
                        if self.freq != "t":
                            df_stamp["minute"] = df_stamp.date.apply(
                                lambda row: row.minute, 1
                            )
                            if self.freq != "s":
                                df_stamp["second"] = df_stamp.date.apply(
                                    lambda row: row.second, 1
                                )
            data_stamp = df_stamp.drop(["date"], axis=1).values
            # TODO：看一下时间戳更细时能不能提取更多数据
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_stamp = pd.DataFrame(data_stamp)
