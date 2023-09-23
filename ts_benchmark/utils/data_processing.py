# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np
import pandas as pd



# def read_data(path: str) -> pd.DataFrame:
#     """
#     读取数据文件并返回 DataFrame。
#
#     根据提供的文件路径，读取数据文件并返回对应的 DataFrame。
#
#     :param path: 数据文件的路径。
#
#     :return: 数据文件内容的 DataFrame。
#     """
#     data = pd.read_csv(path)
#     label_exists = "label" in data["cols"].values
#
#     all_points = data.shape[0]
#     columns = data.columns
#
#     if columns[0] == "date":
#         n_points = data.iloc[:, 2].value_counts().max()
#     else:
#         n_points = data.iloc[:, 1].value_counts().max()
#
#     is_univariate = n_points == all_points
#
#     n_cols = all_points // n_points
#     df = pd.DataFrame()
#
#     if columns[0] == "date" and not is_univariate:
#         df["date"] = data.iloc[:n_points, 0]
#         col_data = {
#             f"col_{j + 1}": data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
#             for j in range(n_cols)
#         }
#         df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
#         if not np.issubdtype(df["date"], np.integer):
#             df["date"] = pd.to_datetime(df["date"])
#         df.set_index("date", inplace=True)

#     elif columns[0] != "date" and not is_univariate:
#         col_data = {
#             f"col_{j + 1}": data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
#             for j in range(n_cols)
#         }
#         df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
#
#     elif columns[0] == "date" and is_univariate:
#         df["date"] = data.iloc[:, 0]
#         df["col_1"] = data.iloc[:, 1]
#         if not np.issubdtype(df["date"], np.integer):
#             df["date"] = pd.to_datetime(df["date"])
#         df.set_index("date", inplace=True)
#
#     else:
#         df["col_1"] = data.iloc[:, 0]
#
#     if label_exists:
#         # 获取最后一列的列名
#         last_col_name = df.columns[-1]
#         # 重新命名最后一列为 "label"
#         df.rename(columns={last_col_name: "label"}, inplace=True)
#
#     return df

def read_data(path: str) -> pd.DataFrame:
    """
    读取数据文件并返回 DataFrame。

    根据提供的文件路径，读取数据文件并返回对应的 DataFrame。

    :param path: 数据文件的路径。

    :return: 数据文件内容的 DataFrame。
    """
    data = pd.read_csv(path)
    label_exists = "label" in data["cols"].values

    all_points = data.shape[0]
    columns = data.columns

    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()

    is_univariate = n_points == all_points

    n_cols = all_points // n_points
    df = pd.DataFrame()

    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        if not np.issubdtype(df["date"], np.integer):
            df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]
        if not np.issubdtype(df["date"], np.integer):
            df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        # 获取最后一列的列名
        last_col_name = df.columns[-1]
        # 重新命名最后一列为 "label"
        df.rename(columns={last_col_name: "label"}, inplace=True)

    return df

def split_before(data: pd.DataFrame, index: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    在指定索引处分割时间序列数据为两部分。

    :param data: 待分割的时间序列数据。
    :param index: 分割索引位置。
    :return: 分割后的前半部分和后半部分数据。
    """
    return data.iloc[:index, :], data.iloc[index:, :]

