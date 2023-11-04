# -*- coding: utf-8 -*-
import concurrent.futures
import os
from scipy.signal import argrelextrema
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stl._stl import STL
import time

import warnings

from ts_benchmark.utils.data_processing import read_data

warnings.filterwarnings("ignore")


def adjust_period(period_value):
    if abs(period_value - 4) <= 1:
        period_value = 4
    if abs(period_value - 7) <= 1:
        period_value = 7
    if abs(period_value - 12) <= 2:
        period_value = 12
    if abs(period_value - 24) <= 3:
        period_value = 24
    if abs(period_value - 48) <= 1 or (
        (48 - period_value) <= 4 and (48 - period_value) >= 0
    ):
        period_value = 48
    if abs(period_value - 52) <= 2:
        period_value = 52
    if abs(period_value - 96) <= 1:
        period_value = 96
    if abs(period_value - 144) <= 4:
        period_value = 144
    if abs(period_value - 168) <= 4:
        period_value = 168
    if abs(period_value - 672) <= 10:
        period_value = 672
    if abs(period_value - 720) <= 25:
        period_value = 720
    return period_value


def fftTransfer(timeseries, fmin=0.2):
    yf = abs(np.fft.fft(timeseries))  # 获取振幅谱
    yfnormlize = yf / len(timeseries)  # 归一化处理
    yfhalf = yfnormlize[: len(timeseries) // 2] * 2  # 由于对称性，只取一半区间

    fwbest = yfhalf[
        argrelextrema(yfhalf, np.greater)
    ]  # 使用 argrelextrema 函数找到频域表示中的局部极大值点（幅度大于其相邻点的点），这些点表示了时间序列中的主要频率成分; fwbest 包含了这些极大值点对应的幅度值。
    xwbest = argrelextrema(yfhalf, np.greater)  # 这里找到了局部极大值点的索引，表示它们在频域表示中的位置。

    fwbest = fwbest[
        fwbest >= fmin
    ].copy()  # 对于满足幅度大于等于 fmin 的极大值点，筛选出它们，然后创建它们的副本，以确保不会改变原始数组。这个步骤是根据 fmin 参数进行的幅度筛选。

    return len(timeseries) / xwbest[0][: len(fwbest)], fwbest  # 返回周期和满足条件的频率分量的幅度值


def count_inversions(series):
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr, 0

        mid = len(arr) // 2
        left, inversions_left = merge_sort(arr[:mid])
        right, inversions_right = merge_sort(arr[mid:])

        merged = []
        inversions = inversions_left + inversions_right

        i, j = 0, 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                inversions += len(left) - i

        merged.extend(left[i:])
        merged.extend(right[j:])

        return merged, inversions

    series_values = series.tolist()
    _, inversions_count = merge_sort(series_values)

    return inversions_count


def count_peaks_and_valleys(sequence):
    peaks = 0
    valleys = 0

    for i in range(1, len(sequence) - 1):
        if sequence[i] > sequence[i - 1] and sequence[i] > sequence[i + 1]:
            peaks += 1
        elif sequence[i] < sequence[i - 1] and sequence[i] < sequence[i + 1]:
            valleys += 1

    return peaks + valleys


def count_series(sequence, threshold):
    if len(sequence) == 0:
        return 0

    # 初始化正类别和负类别的序列数量
    positive_series = 0
    negative_series = 0

    # 标志位，用于跟踪当前子序列的类别
    current_class = None

    for value in sequence:
        if value > threshold:
            # 当前值属于正类别
            if current_class == "negative":
                # 如果之前的子序列是负类别的，增加负类别序列数量
                negative_series += 1
            current_class = "positive"
        else:
            # 当前值属于负类别
            if current_class == "positive":
                # 如果之前的子序列是正类别的，增加正类别序列数量
                positive_series += 1
            current_class = "negative"

    # 最后，根据最后一个子序列的类别增加相应的序列数量
    if current_class == "positive":
        positive_series += 1
    elif current_class == "negative":
        negative_series += 1

    return positive_series + negative_series


def extract_other_features(series_value):
    # 计算偏度
    skewness = skew(series_value)

    # 计算峰度
    kurt = kurtosis(series_value)

    # 计算相对标准偏差（RSD）
    rsd = (np.std(series_value) / np.mean(series_value)) * 100

    # 计算一阶导数的标准差
    std_of_first_derivative = np.std(np.diff(series_value))

    # 计算逆序对的数量
    inversions = count_inversions(series_value) / len(series_value)

    # 计算峰值和低谷总数
    turning_points = count_peaks_and_valleys(series_value) / len(series_value)

    # 计算子序列数量
    series_in_series = count_series(series_value, np.median(series_value)) / len(
        series_value
    )

    return [
        skewness,
        kurt,
        rsd,
        std_of_first_derivative,
        inversions,
        turning_points,
        series_in_series,
    ]


def feature_extract(path):
    index_columns = [
        "file_name",
        "length",
        "period_value1",
        "seasonal_strength1",
        "trend_strength1",
        "period_value2",
        "seasonal_strength2",
        "trend_strength2",
        "period_value3",
        "seasonal_strength3",
        "trend_strength3",
        "if_season",
        "if_trend",
        "ADF:p-value",
        "KPSS:p-value",
        "stability",
        "skewness",
        "kurt",
        "rsd",
        "std_of_first_derivative",
        "inversions",
        "turning_points",
        "series_in_series",
    ]
    result_frame = pd.DataFrame(columns=index_columns)

    file_name = path.split("/")[-1]
    file_name = [file_name]

    # original_df = pd.read_csv(path, header=None)
    # limited_length_df = pd.read_csv(path, header=None, nrows=10000)

    original_df = read_data(path)
    limited_length_df = read_data(path, nrows=10000)


    series_length = [original_df.shape[0]]
    try:
        # ADF Test 原假设是非平稳， P值小于0.05时序列是平稳的， P值越小越平稳。如果p_value值比0.05小，证明有单位根，也就是说序列平稳。如果p_value比0.05大则证明非平稳。
        # ADF_P_value = adfuller(limited_length_df.iloc[:, 0].values, autolag="AIC")[1]
        ADF_P_value = [adfuller(limited_length_df.iloc[:, 0].values + 1e-10, autolag="AIC")[1]]

        # KPSS Test 原假设是平稳的， P值小于0.05则序列是非平稳的， P值越大越平稳
        KPSS_P_value = [kpss(limited_length_df.iloc[:, 0].values, regression="c")[1]]

        stability = [ADF_P_value[0] <= 0.05 or KPSS_P_value[0] >= 0.05]

    except:
        ADF_P_value = [None]
        KPSS_P_value = [None]
        stability = [None]

    series_value = limited_length_df.iloc[:, 0]
    origin_series_value = original_df.iloc[:, 0]
    series_value = pd.Series(series_value).astype("float")
    origin_series_value = pd.Series(origin_series_value).astype("float")
    other_features = extract_other_features(origin_series_value)
    periods, amplitude = fftTransfer(series_value, fmin=0.015)  # 快速傅里叶变换

    periods_list = []
    # 按照振幅大小，保留振幅大的对应的周期值
    for i in range(len(amplitude)):
        periods_list.append(
            round(periods[amplitude.tolist().index(sorted(amplitude, reverse=True)[i])])
        )

    # 筛选后的列表中不包含相同的周期值（去重）且周期值大于或等于4。
    final_periods = []
    for l1 in periods_list:
        if l1 not in final_periods and l1 >= 4:
            final_periods.append(l1)

    yuzhi = int((limited_length_df.shape[0] - 20) / 3)
    if yuzhi <= 12:
        yuzhi = 12

    season_dict = {}
    for i in range(min(10, len(final_periods))):
        period_value = adjust_period(final_periods[i])

        if period_value < yuzhi:
            res = STL(limited_length_df.iloc[:, 0], period=period_value).fit()
            limited_length_df["trend"] = res.trend
            limited_length_df["seasonal"] = res.seasonal
            limited_length_df["resid"] = res.resid
            limited_length_df["detrend"] = (
                limited_length_df.iloc[:, 0] - limited_length_df.trend
            )
            limited_length_df["deseasonal"] = (
                limited_length_df.iloc[:, 0] - limited_length_df.seasonal
            )
            trend_strength = max(
                0,
                1 - limited_length_df.resid.var() / limited_length_df.deseasonal.var(),
            )
            seasonal_strength = max(
                0, 1 - limited_length_df.resid.var() / limited_length_df.detrend.var()
            )
            season_dict[seasonal_strength] = [
                period_value,
                seasonal_strength,
                trend_strength,
            ]

    if len(season_dict) < 3:
        for i in range(3 - len(season_dict)):
            season_dict[0.1 * (i + 1)] = [0, -1, -1]

    season_dict = sorted(season_dict.items(), key=lambda x: x[0], reverse=True)

    result_list = []

    for num, (key, value) in enumerate(season_dict):
        if num == 0:
            max_seasonal_strength = value[1]
            max_trend_strength = value[2]
        if num <= 2:
            result_list = result_list + value

    if_seasonal = [max_seasonal_strength >= 0.9]
    if_trend = [max_trend_strength >= 0.85]
    result_list = (
        file_name
        + series_length
        + result_list
        + if_seasonal
        + if_trend
        + ADF_P_value
        + KPSS_P_value
        + stability
        + other_features
    )

    result_frame.loc[len(result_frame.index)] = result_list
    print(result_list)
    return result_frame


dir_path = os.path.join(r"/Users/xiangfeiqiu/D/datasets/processed_data/forecast_univariate/Libra/transformational_Libra/nature")
file_paths_list = []
for filename in os.listdir(dir_path):
    path = os.path.join(dir_path, filename)
    file_paths_list.append(path)

start_time = time.time()

# 使用多线程并行处理文件
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(feature_extract, path) for path in file_paths_list]

# 使用wait函数等待所有任务完成
concurrent.futures.wait(futures)

# 获取已完成的任务结果
completed_results = [future.result() for future in futures]

# 将所有特征提取结果合并
combined_features = pd.concat(completed_results, ignore_index=True)
end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time
print(f"执行时间为: {execution_time} 秒")

# 显示合并后的特征
print(combined_features)
combined_features.to_csv("/Users/xiangfeiqiu/D/datasets/processed_data/forecast_univariate/Libra/transformational_Libra/nature.csv", index=False)


# # 使用多线程并行处理文件
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     futures = [executor.submit(feature_extract, path) for path in file_paths_list]
#
#     # 从Future对象列表中获取已完成的特征提取结果
#     completed_results = [
#         future.result() for future in concurrent.futures.as_completed(futures)
#     ]
#
# # 将所有特征提取结果合并
# combined_features = pd.concat(completed_results, ignore_index=True)
# end_time = time.time()
#
# # 计算执行时间
# execution_time = end_time - start_time
# print(f"执行时间为: {execution_time} 秒")
#
# combined_features.to_csv(r"/Users/xiangfeiqiu/D/datasets/monash/test.csv", index=False)
# # 显示合并后的特征
# print(combined_features)