# -*- coding: utf-8 -*-
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ts_benchmark.baselines.time_series_library.utils.timefeatures import time_features

from ts_benchmark.utils.data_processing import split_before


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


# class DataloaderForTransformer:
#     def __init__(
#         self,
#         dataset: pd.DataFrame,
#         batch_size: int = 1,
#         history_len: int = 10,
#         prediction_len: int = 2,
#         label_len: int = 5,
#         shuffle: bool = True,
#         timeenc: int = 0,
#         freq: str = "h",
#         seasonal_patterns=None,
#     ):
#         # init
#
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.history_length = history_len
#         self.prediction_length = prediction_len
#         self.label_length = label_len
#         self.shuffle = shuffle
#         self.current_index = 0
#         self.timeenc = timeenc
#         self.freq = freq
#
#         self.__read_data__()
#
#     def __len__(self) -> int:
#         """
#         返回数据加载器的长度。
#
#         :return: 数据加载器的长度。
#         """
#         return len(self.dataset) - self.history_length - self.prediction_length + 1
#
#     def __iter__(self) -> "DataloaderForTransformer":
#         """
#         创建迭代器并返回。
#
#         :return: 数据加载器迭代器。
#         """
#         if self.shuffle:
#             self._shuffle_dataset()
#         self.current_index = 0
#
#         return self
#
#     def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         生成下一个批次的数据。
#
#         :return: 包含输入数据和目标数据的元组。
#         """
#         if self.current_index >= len(self):
#             raise StopIteration
#
#         batch_inputs = []
#         batch_targets = []
#         batch_inputs_mark = []
#         batch_targets_mark = []
#         for _ in range(self.batch_size):
#             # padding_num = self.batch_size - self.dataset.shape[0] % self.batch_size
#             # df = pd.Dataself.dataset.iloc[-1, :] * padding_num
#             if self.current_index + self.history_length + self.prediction_length > len(
#                 self.dataset
#             ):
#                 break
#
#             window_data = self.dataset.iloc[
#                 self.current_index : self.current_index
#                 + self.history_length
#                 + self.prediction_length,
#                 :,
#             ]
#             # if len(window_data) < self.history_length + self.prediction_length:
#             #     raise StopIteration  # 数据集不足一个窗口大小和预测步长时，停止迭代
#
#             inputs = window_data.iloc[: self.history_length].values
#             targets = window_data.iloc[
#                 self.history_length
#                 - self.label_length : self.history_length
#                 + self.prediction_length
#             ].values
#
#             batch_inputs.append(inputs)
#             batch_targets.append(targets)
#
#             window_mark = self.data_stamp.iloc[
#                 self.current_index : self.current_index
#                 + self.history_length
#                 + self.prediction_length,
#             ]
#             inputs_mark = window_mark.iloc[: self.history_length].values
#             targets_mark = window_mark.iloc[
#                 self.history_length
#                 - self.label_length : self.history_length
#                 + self.prediction_length
#             ].values
#             batch_inputs_mark.append(inputs_mark)
#             batch_targets_mark.append(targets_mark)
#             self.current_index += 1
#
#         # 将NumPy数组转换为PyTorch张量
#         batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32)
#         batch_targets = torch.tensor(batch_targets, dtype=torch.float32)
#         batch_inputs_mark = torch.tensor(batch_inputs_mark, dtype=torch.float32)
#         batch_targets_mark = torch.tensor(batch_targets_mark, dtype=torch.float32)
#
#         return batch_inputs, batch_targets, batch_inputs_mark, batch_targets_mark
#
#     def _shuffle_dataset(self):
#         """
#         对数据集进行洗牌。
#         """
#         self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)
#         self.__read_data__()
#
#     def __read_data__(self):
#         df_stamp = self.dataset.reset_index()
#         df_stamp = df_stamp[["date"]]
#         df_stamp["date"] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
#             if self.freq != "m":
#                 df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
#                 if self.freq != "w":
#                     df_stamp["weekday"] = df_stamp.date.apply(
#                         lambda row: row.weekday(), 1
#                     )
#                     if self.freq != "b" and self.freq != "d":
#                         df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
#                         if self.freq != "h":
#                             df_stamp["minute"] = df_stamp.date.apply(
#                                 lambda row: row.minute, 1
#                             )
#                             if self.freq != "t":
#                                 df_stamp["second"] = df_stamp.date.apply(
#                                     lambda row: row.minute, 1
#                                 )
#
#             data_stamp = df_stamp.drop(["date"], axis=1).values
#             # TODO：看一下时间戳更细时能不能提取更多数据
#         elif self.timeenc == 1:
#             data_stamp = time_features(
#                 pd.to_datetime(df_stamp["date"].values), freq=self.freq
#             )
#             data_stamp = data_stamp.transpose(1, 0)
#
#         self.data_stamp = pd.DataFrame(data_stamp)


def train_val_split(train_data, ratio, seq_len):
    border = int((train_data.shape[0]) * ratio)

    train_data_value, valid_data_rest = split_before(train_data, border)
    train_data_rest, valid_data = split_before(train_data, border - seq_len)
    return train_data_value, valid_data


def data_provider(data, config, timeenc, batch_size, shuffle, drop_last):
    dataset = DatasetForTransformer(
        dataset=data,
        history_len=config.seq_len,
        prediction_len=config.pred_len,
        label_len=config.label_len,
        timeenc=timeenc,
        freq=config.freq,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=drop_last,
    )

    return dataset, data_loader


class DatasetForTransformer:
    def __init__(
        self,
        dataset: pd.DataFrame,
        history_len: int = 10,
        prediction_len: int = 2,
        label_len: int = 5,
        timeenc: int = 1,
        freq: str = "h",
    ):
        # init

        self.dataset = dataset
        self.history_length = history_len
        self.prediction_length = prediction_len
        self.label_length = label_len
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
                        if self.freq != "h":
                            df_stamp["minute"] = df_stamp.date.apply(
                                lambda row: row.minute, 1
                            )
                            if self.freq != "t":
                                df_stamp["second"] = df_stamp.date.apply(
                                    # lambda row: row.minute, 1
                                    lambda row: row.second,
                                    1,
                                )

            data_stamp = df_stamp.drop(["date"], axis=1).values
            # TODO：看一下时间戳更细时能不能提取更多数据
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_stamp = pd.DataFrame(data_stamp, dtype=float)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.history_length
        r_begin = s_end - self.label_length
        r_end = r_begin + self.label_length + self.prediction_length

        seq_x = self.dataset[s_begin:s_end]
        seq_y = self.dataset[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = torch.tensor(seq_x.values, dtype=torch.float32)
        seq_y = torch.tensor(seq_y.values, dtype=torch.float32)
        seq_x_mark = torch.tensor(seq_x_mark.values, dtype=torch.float32)
        seq_y_mark = torch.tensor(seq_y_mark.values, dtype=torch.float32)
        return seq_x, seq_y, seq_x_mark, seq_y_mark
