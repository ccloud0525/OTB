import os.path
import time
from datetime import datetime, timedelta
from distutils.util import strtobool

import numpy as np
import pandas as pd
import tqdm
from dateutil.relativedelta import relativedelta

from ts_benchmark.common.constant import ROOT_PATH

# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the data_loader: frequency, horizon, whether the data_loader contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
from ts_benchmark.utils.data_processing import read_data
from ts_benchmark.utils.parallel import ParallelBackend


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)
        loaded_data = loaded_data.iloc[:, 1:50]
        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def convert_monash_multivariate(path, root_path, info):
    (
        loaded_data,
        frequency,
        forecast_horizon,
        contain_missing_values,
        contain_equal_length,
    ) = convert_tsf_to_dataframe(path)
    if forecast_horizon is None:
        forecast_horizon = info[5]

    index_columns = [
        "dataset_name",
        "file_name",
        "domain",
        "is_univariate",
        "has_timestamp",
        "rugular",
        "freq",
        "forecast_horizon",
        "abnormal_ratio",
        "train_lens",
        "val_lens",
        "train_has_label",
        "val_has_label",
        "data_sources",
        "licence",
        "other_situation",
        "contain_missing_values",
    ]
    other_info_df = pd.DataFrame(columns=index_columns)

    has_date = info[0]
    value_index = info[1]
    time_index = info[2]
    frequency_dict = {
        "daily": "days",
        "weekly": "weeks",
        "monthly": "months",
        "yearly": "years",
        "hourly": "hours",
    }

    n_cols = loaded_data.iloc[:, time_index].shape[0]
    time_points = loaded_data.iloc[0, value_index].shape[0]

    log_file_name = path.split("/")[-1].split(".")[0] + "log.txt"
    with open(
        os.path.join(
            "/Users/xiangfeiqiu/D/datasets/多元时序预测数据集汇总/monash_log", log_file_name
        ),
        "w",
    ) as log_file:
        # for i in range(n_cols):
        # need_data = loaded_data.iloc[i, value_index]
        # # if np.var(need_data) == 0:
        # #     log_message = f"{path}中第 {i+1}条序列方差为0"
        # #     print(log_message)  # 也可以将日志输出到控制台
        # #     log_file.write(log_message + "\n")
        # #     continue
        # df = pd.DataFrame()
        num = loaded_data.iloc[0, value_index].shape[0]

        # pd.offsets.QuarterEnd()

        if has_date:
            start_time = loaded_data.iloc[0, time_index]
            try:
                if frequency == "quarterly":
                    date_range = [
                        (start_time + pd.DateOffset(months=i * 3)).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        for i in range(num)
                    ]
                elif frequency == "half_hourly":
                    date_range = [
                        (start_time + pd.DateOffset(minutes=i * 30)).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        for i in range(num)
                    ]
                else:
                    date_range = [
                        (
                            start_time + pd.DateOffset(**{frequency_dict[frequency]: i})
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        for i in range(num)
                    ]
            except Exception as e:
                date_range = list(range(1, num + 1))
                # log_message = f"{path}中第{i + 1}条序列超出时间戳"
                # print(log_message)  # 也可以将日志输出到控制台
                # log_file.write(log_message + "\n")
        else:
            date_range = list(range(1, num + 1))

    data_list = []
    cols_list = []
    file_name = path.split("/")[-1].split(".")[0]
    new_file_name = file_name + ".csv"
    name = [file_name + ";" + new_file_name]

    for i in range(n_cols):
        data_list = data_list + loaded_data.iloc[i, value_index].tolist()
        col_mame = "channel" + "_" + str(i + 1)
        col = [col_mame] * time_points
        cols_list = cols_list + col

    df_new = pd.DataFrame()
    df_new.insert(0, "date", date_range * n_cols)
    df_new.insert(1, "data", data_list)
    df_new.insert(2, "cols", cols_list)
    df_new.insert(3, "name", name * n_cols * time_points)

    path = os.path.join(root_path, new_file_name)
    df_new.to_csv(path, index=False)

    other_info_df.loc[len(other_info_df.index)] = (
        [file_name]
        + [new_file_name]
        + [info[2]]
        + [False]
        + [has_date]
        + [True]
        + [frequency]
        + [forecast_horizon]
        + [np.nan]
        + [np.nan]
        + [np.nan]
        + [np.nan]
        + [np.nan]
        + [info[2]]
        + ["CC BY 4.0 DEED"]
        + [np.nan]
        + [contain_missing_values]
    )
    info_file_name = file_name + "_info.csv"
    other_info_df.to_csv(
        os.path.join(
            "/Users/xiangfeiqiu/D/datasets/多元时序预测数据集汇总/",
            "monash_other_info",
            info_file_name,
        ),
        index=False,
    )

    return other_info_df


# has_date, value_index
monash_dict = {
    "car_parts_dataset_without_missing_values": (
        True,
        1,
        0,
        ["Sales"],
        ["real; OpenSource"],
        12,
    ),
    "covid_deaths_dataset": (True, 1, 0, ["Nature"], ["real; OpenSource"], 30),
    "electricity_hourly_dataset": (True, 1, 0, ["Energy"], ["real; OpenSource"], 168),
    "electricity_weekly_dataset": (True, 1, 0, ["Energy"], ["real; OpenSource"], 8),
    "fred_md_dataset": (True, 1, 0, ["Economic"], ["real; OpenSource"], 12),
    "hospital_dataset": (True, 1, 0, ["Health"], ["real; OpenSource"], 12),
    # "kaggle_web_traffic_weekly_dataset": (True, 1, 0, ["Web"], ["real; OpenSource"], 8),
    "nn5_daily_dataset_without_missing_values": (
        True,
        1,
        0,
        ["Banking"],
        ["real; OpenSource"],
        np.nan,
    ),
    "nn5_weekly_dataset": (True, 1, 0, ["Banking"], ["real; OpenSource"], 8),
    "rideshare_dataset_without_missing_values": (
        True,
        5,
        4,
        ["Transport"],
        ["real; Curated_by_monash"],
        168,
    ),
    "solar_10_minutes_dataset": (True, 1, 0, ["Energy"], ["real; OpenSource"], 1008),
    "solar_weekly_dataset": (True, 1, 0, ["Energy"], ["real; OpenSource"], 5),
    # "temperature_rain_dataset_without_missing_values": (True, 3, 2, ["Nature"], ["real; Curated_by_monash"], 30),
    "traffic_weekly_dataset": (True, 1, 0, ["Transport"], ["real; OpenSource"], 8),
    "traffic_hourly_dataset": (True, 1, 0, ["Transport"], ["real; OpenSource"], 168),
    "australian_electricity_demand_dataset": (
        True,
        2,
        1,
        ["Energy"],
        ["real; Curated_by_monash"],
    ),
    "bitcoin_dataset_without_missing_values": (
        True,
        1,
        0,
        ["Economic"],
        ["real; Curated_by_monash"],
    ),
    "cif_2016_dataset": (False, 1, np.nan, ["Banking"], ["real; OpenSource"]),
    "kdd_cup_2018_dataset_without_missing_values": (
        True,
        4,
        3,
        ["Nature"],
        ["real; OpenSource"],
    ),
    "m1_monthly_dataset": (True, 1, 0, ["Multiple"], ["real; OpenSource"]),
    "m1_quarterly_dataset": (True, 1, 0, ["Multiple"], ["real; OpenSource"]),
    "m1_yearly_dataset": (True, 1, 0, ["Multiple"], ["real; OpenSource"]),
    "m3_monthly_dataset": (True, 1, 0, ["Multiple"], ["real; OpenSource"]),
    "m3_other_dataset": (False, 0, np.nan, ["Multiple"], ["real; OpenSource"]),
    "m3_quarterly_dataset": (True, 1, 0, ["Multiple"], ["real; OpenSource"]),
    "m3_yearly_dataset": (True, 1, 0, ["Multiple"], ["real; OpenSource"]),
    "m4_daily_dataset": (True, 1, 0, ["Multiple"], ["real; OpenSource"]),
    "m4_hourly_dataset": (True, 1, 0, ["Multiple"], ["real; OpenSource"]),
    "m4_monthly_dataset": (True, 1, 0, ["Multiple"], ["real; OpenSource"]),
    "m4_quarterly_dataset": (True, 1, 0, ["Multiple"], ["real; OpenSource"]),
    "m4_weekly_dataset": (True, 1, 0, ["Multiple"], ["real; OpenSource"]),
    "m4_yearly_dataset": (True, 1, 0, ["Multiple"], ["real; OpenSource"]),
    "pedestrian_counts_dataset": (True, 1, 0, ["Transport"], ["real; OpenSource"]),
    "saugeenday_dataset": (True, 1, 0, ["Nature"], ["real; OpenSource"]),
    "sunspot_dataset_without_missing_values": (
        True,
        1,
        0,
        ["Nature"],
        ["real; OpenSource"],
    ),
    "tourism_monthly_dataset": (True, 1, 0, ["Tourism"], ["real; OpenSource"]),
    "tourism_quarterly_dataset": (True, 1, 0, ["Tourism"], ["real; OpenSource"]),
    "tourism_yearly_dataset": (True, 1, 0, ["Tourism"], ["real; OpenSource"]),
    "us_births_dataset": (True, 1, 0, ["Nature"], ["real; OpenSource"]),
    "vehicle_trips_dataset_without_missing_values": (
        True,
        4,
        3,
        ["Transport"],
        ["real; OpenSource"],
    ),
    "weather_dataset": (False, 1, np.nan, ["Nature"], ["real; OpenSource"]),
    "dominick_dataset": (False, 0, np.nan, ["Sales"], ["real; OpenSource"]),
    # "kaggle_web_traffic_dataset_without_missing_values": (True, 1, 0, ["Web"], ["real; OpenSource"], np.nan),
}


def ray_monash_multi():
    ParallelBackend().init(
        backend="ray",
        n_workers=os.cpu_count(),
        n_cpus=os.cpu_count(),
        gpu_devices=None,
        default_timeout=60000,
    )

    eval_backend = ParallelBackend()
    ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", ".."))

    result_list = []
    start_time = time.time()
    # for filename in os.listdir(
    #     os.path.join(ROOT_PATH, "origin_monash")
    # ):
    for filename in os.listdir("/Users/xiangfeiqiu/D/datasets/多元时序预测数据集汇总/monash"):
        if filename == ".DS_Store":
            continue
        path = os.path.join(
            "/Users/xiangfeiqiu/D/datasets/多元时序预测数据集汇总/monash", filename
        )
        root_path = os.path.join(
            "/Users/xiangfeiqiu/D/datasets/多元时序预测数据集汇总/", "transformational_monash"
        )
        try:
            info = monash_dict[filename.split(".")[0]]
            result_list.append(
                eval_backend.schedule(
                    convert_monash_multivariate, args=(path, root_path, info)
                )
            )
        except Exception as e:
            print(f"{path} is not tested. Exception: {str(e)}")

    # 获取已完成的任务结果
    completed_results = [res.result() for res in result_list]
    # 将所有特征提取结果合并
    combined_features = pd.concat(completed_results, ignore_index=True)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"执行时间为: {execution_time} 秒")
    # 显示合并后的特征
    print(combined_features)

    combined_features.to_csv(
        os.path.join(
            "/Users/xiangfeiqiu/D/datasets/多元时序预测数据集汇总/", "monash_mtea_info.csv"
        ),
        index=False,
    )
    ParallelBackend().close(force=True)


ROOT_PATH = "/Users/xiangfeiqiu/D/datasets/processed_data/forecast_univariate/test"


def convert_monash_univariate(path, root_path, info):
    (
        loaded_data,
        frequency,
        forecast_horizon,
        contain_missing_values,
        contain_equal_length,
    ) = convert_tsf_to_dataframe(path)

    index_columns = [
        "dataset_name",
        "file_name",
        "domain",
        "is_univariate",
        "has_timestamp",
        "rugular",
        "freq",
        "forecast_horizon",
        "abnormal_ratio",
        "train_lens",
        "val_lens",
        "train_has_label",
        "val_has_label",
        "data_sources",
        "licence",
        "other_situation",
        "contain_missing_values",
        "contain_equal_length",
    ]
    other_info_df = pd.DataFrame(columns=index_columns)

    has_date = info[0]
    value_index = info[1]
    time_index = info[2]
    frequency_dict = {
        "daily": "days",
        "weekly": "weeks",
        "monthly": "months",
        "yearly": "years",
        "hourly": "hours",
    }
    log_file_name = path.split("/")[-1].split(".")[0] + "log.txt"
    directory = os.path.dirname(os.path.join(ROOT_PATH, "monash_log", log_file_name))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(ROOT_PATH, "monash_log", log_file_name), "w") as log_file:
        for i in range(loaded_data.shape[0]):
            need_data = loaded_data.iloc[i, value_index]
            if np.var(need_data) == 0:
                log_message = f"{path}中第 {i+1}条序列方差为0"
                # print(log_message)  # 也可以将日志输出到控制台
                log_file.write(log_message + "\n")
                log_file.flush()  # 实时刷新到磁盘
                continue
            df = pd.DataFrame()
            num = loaded_data.iloc[i, value_index].shape[0]

            if has_date:
                start_time = loaded_data.iloc[i, time_index]
                # try:
                #     if frequency == "quarterly":
                #         date_range = [
                #             (start_time + pd.DateOffset(months=i * 3)).strftime(
                #                 "%Y-%m-%d %H:%M:%S"
                #             )
                #             for i in range(num)
                #         ]
                #     elif frequency == "half_hourly":
                #         date_range = [
                #             (start_time + pd.DateOffset(minutes=i * 30)).strftime(
                #                 "%Y-%m-%d %H:%M:%S"
                #             )
                #             for i in range(num)
                #         ]
                try:
                    if frequency == "quarterly":
                        timestamp = pd.Timestamp(start_time)
                        if timestamp.is_month_start:
                            date_range = [
                                (start_time + pd.offsets.MonthBegin() * 3 * i).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                                for i in range(num)
                            ]
                        else:
                            date_range = [
                                (start_time + pd.offsets.MonthEnd() * 3 * i).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                                for i in range(num)
                            ]
                    elif frequency == "half_hourly":
                        date_range = [
                            (start_time + pd.DateOffset(minutes=i * 30)).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                            for i in range(num)
                        ]
                    elif frequency == "monthly":
                        timestamp = pd.Timestamp(start_time)
                        if timestamp.is_month_start:
                            date_range = [
                                (start_time + pd.offsets.MonthBegin() * i).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                                for i in range(num)
                            ]
                        elif timestamp.is_month_end:
                            date_range = [
                                (start_time + pd.offsets.MonthEnd() * i).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                                for i in range(num)
                            ]
                        else:
                            date_range = [
                                (
                                    start_time
                                    + pd.DateOffset(**{frequency_dict[frequency]: i})
                                ).strftime("%Y-%m-%d %H:%M:%S")
                                for i in range(num)
                            ]

                    else:
                        date_range = [
                            (
                                start_time
                                + pd.DateOffset(**{frequency_dict[frequency]: i})
                            ).strftime("%Y-%m-%d %H:%M:%S")
                            for i in range(num)
                        ]
                except Exception as e:
                    date_range = list(range(1, num + 1))
                    log_message = f"{path}中第{i + 1}条序列超出时间戳"
                    # print(log_message)  # 也可以将日志输出到控制台
                    log_file.write(log_message + "\n")
                    log_file.flush()  # 实时刷新到磁盘
            else:
                date_range = list(range(1, num + 1))

            file_name = path.split("/")[-1].split(".")[0]
            new_file_name = file_name + "_" + str(i + 1) + ".csv"
            name = [file_name + ";" + new_file_name] * num
            cols = ["channel_1"] * num
            df.insert(0, "date", date_range)
            df.insert(1, "data", loaded_data.iloc[i, value_index])
            df.insert(2, "cols", cols)
            df.insert(3, "name", name)
            file_path = os.path.join(root_path, new_file_name)
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            df.to_csv(file_path, index=False)
            converted_data = read_data(file_path)
            frq_test = converted_data.index.inferred_freq
            if frq_test is None:
                print(file_path)
                os.remove(file_path)
                log_file.write(file_path + "\n")
                log_file.flush()  # 实时刷新到磁盘

            if file_name == "cif_2016_dataset":
                forecast_horizon = loaded_data.iloc[i, 0]
            other_info_df.loc[len(other_info_df.index)] = (
                [file_name]
                + [new_file_name]
                + info[3]
                + [True]
                + [has_date]
                + [True]
                + [frequency]
                + [forecast_horizon]
                + [np.nan]
                + [np.nan]
                + [np.nan]
                + [np.nan]
                + [np.nan]
                + info[4]
                + ["CC BY 4.0 DEED"]
                + [np.nan]
                + [contain_missing_values]
                + [contain_equal_length]
            )
    info_file_name = file_name + "_info.csv"
    directory = os.path.dirname(
        os.path.join(ROOT_PATH, "monash_other_info", info_file_name)
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    other_info_df.to_csv(
        os.path.join(ROOT_PATH, "monash_other_info", info_file_name), index=False
    )

    return other_info_df


def ray_monash_univariate():
    ParallelBackend().init(
        backend="ray",
        n_workers=os.cpu_count(),
        n_cpus=os.cpu_count(),
        gpu_devices=None,
        default_timeout=60000,
    )

    eval_backend = ParallelBackend()
    # ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", ".."))
    ROOT_PATH = "/Users/xiangfeiqiu/D/datasets/processed_data/forecast_univariate/test"
    result_list = []
    start_time = time.time()
    for filename in os.listdir(os.path.join(ROOT_PATH, "origin_monash")):
        if filename == ".DS_Store":
            continue
        file = filename.split(".")[0]
        path = os.path.join(ROOT_PATH, "origin_monash", filename)
        root_path = os.path.join(ROOT_PATH, "transformational_monash", file)
        info = monash_dict[file]
        try:
            result_list.append(
                eval_backend.schedule(
                    convert_monash_univariate, args=(path, root_path, info)
                )
            )
        except Exception as e:
            print(f"{path} is not tested. Exception: {str(e)}")

    # 获取已完成的任务结果
    completed_results = [res.result() for res in result_list]
    # 将所有特征提取结果合并
    combined_features = pd.concat(completed_results, ignore_index=True)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"执行时间为: {execution_time} 秒")
    # 显示合并后的特征
    print(combined_features)

    combined_features.to_csv(
        os.path.join(ROOT_PATH, "monash_other_info.csv"),
        index=False,
    )
    ParallelBackend().close(force=True)


# ray_monash_univariate()

# def covert_tsb_uad_univarite_series(path):
#     raw_data = pd.read_csv(path, header=None)
#     series_len = len(raw_data)
#     data = pd.concat([raw_data.iloc[:, 0], raw_data.iloc[:, 1]], ignore_index=True)
#     col_name = ["col1"] * series_len + ["label"] * series_len
#     new_df = pd.DataFrame({"data": data, "cols": col_name})
#     file_name = os.path.basename(path)
#     new_file_name = os.path.splitext(file_name)[0] + ".csv"
#     file_path = os.path.join(r"C:\Users\86188\Desktop", new_file_name)
#     new_df.to_csv(file_path, index=False)
#     print(new_df)

tsb_uad_dict = {
    "Dodgers": (["Transport"], ["real; OpenSource"]),
    "ECG": (["Health"], ["real; OpenSource"]),
    "IOPS": (["Web"], ["real; OpenSource"]),
    "KDD21": (["Multiple"], ["real; OpenSource"]),
    "MGAB": (["Multiple"], ["real; OpenSource"]),
    "NAB": (["Multiple"], ["real; OpenSource"]),
    "SensorScope": (["Nature"], ["real; OpenSource"]),
    "YAHOO": (["machinery"], ["real; OpenSource"]),
    "NASA-MSL-new": (["aviation"], ["transformational"]),
    "NASA-SMAP-new": (["aviation"], ["transformational"]),
    "Daphnet": (["Health"], ["transformational"]),
    "GHL": (["machinery"], ["transformational"]),
    "Genesis": (["Multiple"], ["transformational"]),
    "MITDB": (["Health"], ["transformational"]),
    "OPPORTUNITY": (["Sports"], ["transformational"]),
    "Occupancy": (["Nature"], ["transformational"]),
    "SMD": (["Server"], ["transformational"]),
    "SVDB": (["Health"], ["transformational"]),
}


def covert_tsb_uad_univarite_series(path, root_path, info):
    index_columns = [
        "dataset_name",
        "file_name",
        "domain",
        "is_univariate",
        "has_timestamp",
        "rugular",
        "freq",
        "forecast_horizon",
        "abnormal_ratio",
        "train_lens",
        "val_lens",
        "train_has_label",
        "val_has_label",
        "data_sources",
        "licence",
        "other_situation",
        "contain_missing_values",
    ]
    other_info_df = pd.DataFrame(columns=index_columns)
    log_file_name = path.split("/")[-1].split(".")[0] + "_log.txt"
    if not os.path.exists(os.path.join(root_path, "tsb_uad_log")):
        os.makedirs(os.path.join(root_path, "tsb_uad_log"))
    with open(os.path.join(root_path, "tsb_uad_log", log_file_name), "w") as log_file:
        specific = (
            path.split("/")[-1] == "NASA-MSL-new"
            or path.split("/")[-1] == "NASA-SMAP-new"
        )
        for index, file_name in enumerate(os.listdir(path)):
            if file_name == ".DS_Store":
                continue
            dir_name = "tsb_uad_" + path.split("/")[-1]
            new_file_name = file_name
            raw_data = pd.read_csv(os.path.join(path, new_file_name), header=None)
            new_file_name = file_name.rsplit(".", 1)[0] + ".csv"

            series_len = len(raw_data)
            count_of_ones = (raw_data.iloc[:, 1] != 0).sum()
            count_of_zeros = (raw_data.iloc[:, 1] == 0).sum()
            if count_of_zeros == series_len:
                log_message = f"{path}中{new_file_name}没有异常"
                print(log_message)  # 也可以将日志输出到控制台
                log_file.write(log_message + "\n")
                continue
            ratio = float(count_of_ones / series_len)
            data = pd.concat(
                [raw_data.iloc[:, 0], raw_data.iloc[:, 1]], ignore_index=True
            )
            col_name = ["channel_1"] * series_len + ["label"] * series_len
            name = [dir_name + ";" + new_file_name] * series_len
            date_range = list(range(1, series_len + 1)) * 2

            new_df = pd.DataFrame(
                {"date": date_range, "data": data, "cols": col_name, "name": name * 2}
            )
            file_path = os.path.join(
                root_path,
                "transformational_tsb_uad",
                path.split("/")[-1],
                new_file_name,
            )
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            new_df.to_csv(file_path, index=False)
            if specific:
                other_info_df.loc[len(other_info_df.index)] = (
                    [dir_name]
                    + [new_file_name]
                    + info[0]
                    + [True]
                    + [False]
                    + [True]
                    + [np.nan]
                    + [np.nan]
                    + [ratio]
                    + [new_file_name.split("_")[1].split(".")[0]]
                    + [np.nan]
                    + [True]
                    + [np.nan]
                    + info[1]
                    + ["BY-NC-ND 4.0"]
                    + [np.nan]
                    + [False]
                )
            else:
                other_info_df.loc[len(other_info_df.index)] = (
                    [dir_name]
                    + [new_file_name]
                    + info[0]
                    + [True]
                    + [False]
                    + [True]
                    + [np.nan]
                    + [np.nan]
                    + [ratio]
                    + [np.nan]
                    + [np.nan]
                    + [np.nan]
                    + [np.nan]
                    + info[1]
                    + ["BY-NC-ND 4.0"]
                    + [np.nan]
                    + [False]
                )
    info_path = path.split("/")[-1] + "_meta_info.csv"
    if not os.path.exists(os.path.join(root_path, "tsb_uad_meta_info")):
        os.makedirs(os.path.join(root_path, "tsb_uad_meta_info"))
    other_info_df.to_csv(
        os.path.join(root_path, "tsb_uad_meta_info", info_path), index=False
    )

    return other_info_df


def ray_tsb_uad_univariate():
    ParallelBackend().init(
        backend="ray",
        n_workers=os.cpu_count(),
        n_cpus=os.cpu_count(),
        gpu_devices=None,
        default_timeout=60000,
    )

    eval_backend = ParallelBackend()
    ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", ".."))

    result_list = []
    start_time = time.time()

    for filename in os.listdir(
        "/Users/xiangfeiqiu/D/datasets/processed_data/ad_univariate/tsb_uad/origin_tsb_uad"
    ):
        if filename == ".DS_Store":
            continue
        path = os.path.join(
            "/Users/xiangfeiqiu/D/datasets/processed_data/ad_univariate/tsb_uad/origin_tsb_uad",
            filename,
        )
        root_path = (
            "/Users/xiangfeiqiu/D/datasets/processed_data/ad_univariate/tsb_uad/"
        )

        try:
            info = tsb_uad_dict[filename]
            result_list.append(
                eval_backend.schedule(
                    covert_tsb_uad_univarite_series, args=(path, root_path, info)
                )
            )
        except Exception as e:
            print(f"{path} is not tested. Exception: {str(e)}")

    # 获取已完成的任务结果
    completed_results = [res.result() for res in result_list]
    # 将所有特征提取结果合并
    combined_features = pd.concat(completed_results, ignore_index=True)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"执行时间为: {execution_time} 秒")
    # 显示合并后的特征
    print(combined_features)

    combined_features.to_csv(
        os.path.join(
            "/Users/xiangfeiqiu/D/datasets/processed_data/ad_univariate/tsb_uad/",
            "tsb_uad_mtea_info.csv",
        ),
        index=False,
    )
    ParallelBackend().close(force=True)


def covert_zhao_univariate(path, root_path):
    index_columns = [
        "dataset_name",
        "file_name",
        "domain",
        "is_univariate",
        "has_timestamp",
        "rugular",
        "freq",
        "forecast_horizon",
        "abnormal_ratio",
        "train_lens",
        "val_lens",
        "train_has_label",
        "val_has_label",
        "data_sources",
        "licence",
        "other_situation",
        "contain_missing_values",
    ]
    other_info_df = pd.DataFrame(columns=index_columns)
    log_file_name = path.split("/")[-1].split(".")[0] + "_log.txt"
    if not os.path.exists(os.path.join(root_path, "tsb_uad_log")):
        os.makedirs(os.path.join(root_path, "tsb_uad_log"))
    with open(os.path.join(root_path, "zhao_log", log_file_name), "w") as log_file:
        for index, file_name in enumerate(os.listdir(path)):
            if file_name == ".DS_Store":
                continue
            dir_name = path.split("/")[-1]
            new_file_name = file_name
            raw_data = pd.read_csv(os.path.join(path, new_file_name), header=0)
            new_file_name = file_name.rsplit(".", 1)[0] + ".csv"

            series_len = len(raw_data)
            count_of_ones = (raw_data.iloc[:, 1] != 0).sum()
            count_of_zeros = (raw_data.iloc[:, 1] == 0).sum()
            if count_of_zeros == series_len:
                log_message = f"{path}中{new_file_name}没有异常"
                print(log_message)  # 也可以将日志输出到控制台
                log_file.write(log_message + "\n")
                continue
            ratio = float(count_of_ones / series_len)
            data = pd.concat(
                [raw_data.iloc[:, 0], raw_data.iloc[:, 1]], ignore_index=True
            )
            col_name = ["channel_1"] * series_len + ["label"] * series_len
            name = [dir_name + ";" + new_file_name] * series_len
            date_range = list(range(1, series_len + 1)) * 2

            new_df = pd.DataFrame(
                {"date": date_range, "data": data, "cols": col_name, "name": name * 2}
            )
            file_path = os.path.join(
                root_path, "transformational_zhao", path.split("/")[-1], new_file_name
            )
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            new_df.to_csv(file_path, index=False)
            other_info_df.loc[len(other_info_df.index)] = (
                [dir_name]
                + [new_file_name]
                + ["Multiple"]
                + [True]
                + [False]
                + [True]
                + [np.nan]
                + [np.nan]
                + [ratio]
                + [np.nan]
                + [np.nan]
                + [np.nan]
                + [np.nan]
                + ["synthetic"]
                + ["Apache License 2.0"]
                + [np.nan]
                + [False]
            )
    info_path = path.split("/")[-1] + "_meta_info.csv"
    if not os.path.exists(os.path.join(root_path, "zhao_meta_info")):
        os.makedirs(os.path.join(root_path, "zhao_meta_info"))
    other_info_df.to_csv(
        os.path.join(root_path, "zhao_meta_info", info_path), index=False
    )

    return other_info_df


def ray_zhao_univariate():
    ParallelBackend().init(
        backend="ray",
        n_workers=os.cpu_count(),
        n_cpus=os.cpu_count(),
        gpu_devices=None,
        default_timeout=60000,
    )

    eval_backend = ParallelBackend()
    ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", ".."))

    result_list = []
    start_time = time.time()

    for filename in os.listdir(
        "/Users/xiangfeiqiu/D/datasets/processed_data/ad_univariate/zhaoyue/"
    ):
        if filename == ".DS_Store":
            continue
        path = os.path.join(
            "/Users/xiangfeiqiu/D/datasets/processed_data/ad_univariate/zhaoyue/",
            filename,
        )
        root_path = (
            "/Users/xiangfeiqiu/D/datasets/processed_data/ad_univariate/zhaoyue/"
        )

        try:
            result_list.append(
                eval_backend.schedule(covert_zhao_univariate, args=(path, root_path))
            )
        except Exception as e:
            print(f"{path} is not tested. Exception: {str(e)}")

    # 获取已完成的任务结果
    completed_results = [res.result() for res in result_list]
    # 将所有特征提取结果合并
    combined_features = pd.concat(completed_results, ignore_index=True)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"执行时间为: {execution_time} 秒")
    # 显示合并后的特征
    print(combined_features)

    combined_features.to_csv(
        os.path.join(
            "/Users/xiangfeiqiu/D/datasets/processed_data/ad_univariate/zhaoyue/",
            "zhao_mtea_info.csv",
        ),
        index=False,
    )
    ParallelBackend().close(force=True)


def covert_libra(path, root_path):
    index_columns = [
        "dataset_name",
        "file_name",
        "domain",
        "is_univariate",
        "has_timestamp",
        "rugular",
        "freq",
        "forecast_horizon",
        "abnormal_ratio",
        "train_lens",
        "val_lens",
        "train_has_label",
        "val_has_label",
        "data_sources",
        "licence",
        "other_situation",
        "contain_missing_values",
    ]
    other_info_df = pd.DataFrame(columns=index_columns)

    for index, file_name in enumerate(os.listdir(path)):
        if file_name == ".DS_Store":
            continue
        dir_name = "Libra_" + path.split("/")[-1]
        new_file_name = file_name
        raw_data = pd.read_csv(os.path.join(path, new_file_name), header=None)
        series_len = len(raw_data)
        data = raw_data.iloc[:, 0]
        name = [dir_name + ";" + new_file_name] * series_len
        col_name = ["channel_1"] * series_len
        date_range = list(range(1, series_len + 1))

        new_df = pd.DataFrame(
            {"date": date_range, "data": data, "cols": col_name, "name": name}
        )
        file_path = os.path.join(root_path, path.split("/")[-1], new_file_name)
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        new_df.to_csv(file_path, index=False)
        other_info_df.loc[len(other_info_df.index)] = (
            [dir_name]
            + [new_file_name]
            + [path.split("/")[-1]]
            + [True]
            + [False]
            + [True]
            + [np.nan]
            + [np.nan]
            + [np.nan]
            + [np.nan]
            + [np.nan]
            + [np.nan]
            + [np.nan]
            + ["real"]
            + ["GNU General Public License v3.0"]
            + [np.nan]
            + [False]
        )
    info_path = path.split("/")[-1] + "_meta_info.csv"
    if not os.path.exists(
        os.path.join(
            "/Users/xiangfeiqiu/D/datasets/processed_data/forecast_univariate/Libra",
            "Libra_meta_info",
        )
    ):
        os.makedirs(
            os.path.join(
                "/Users/xiangfeiqiu/D/datasets/processed_data/forecast_univariate/Libra",
                "Libra_meta_info",
            )
        )
    other_info_df.to_csv(
        os.path.join(
            "/Users/xiangfeiqiu/D/datasets/processed_data/forecast_univariate/Libra",
            "Libra_meta_info",
            info_path,
        ),
        index=False,
    )

    return other_info_df


def ray_libra_univariate():
    ParallelBackend().init(
        backend="ray",
        n_workers=os.cpu_count(),
        n_cpus=os.cpu_count(),
        gpu_devices=None,
        default_timeout=60000,
    )

    eval_backend = ParallelBackend()
    ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", ".."))

    result_list = []
    start_time = time.time()
    # for filename in os.listdir(
    #     os.path.join(ROOT_PATH, "origin_monash")
    # ):
    for filename in os.listdir(
        "/Users/xiangfeiqiu/D/datasets/processed_data/forecast_univariate/Libra/origin_data"
    ):
        if filename == ".DS_Store":
            continue
        path = os.path.join(
            "/Users/xiangfeiqiu/D/datasets/processed_data/forecast_univariate/Libra/origin_data",
            filename,
        )
        root_path = os.path.join(
            "/Users/xiangfeiqiu/D/datasets/processed_data/forecast_univariate/Libra",
            "transformational_Libra",
        )
        try:
            result_list.append(
                eval_backend.schedule(covert_libra, args=(path, root_path))
            )
        except Exception as e:
            print(f"{path} is not tested. Exception: {str(e)}")

    # 获取已完成的任务结果
    completed_results = [res.result() for res in result_list]
    # 将所有特征提取结果合并
    combined_features = pd.concat(completed_results, ignore_index=True)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"执行时间为: {execution_time} 秒")
    # 显示合并后的特征
    print(combined_features)

    combined_features.to_csv(
        os.path.join(
            "/Users/xiangfeiqiu/D/datasets/processed_data/forecast_univariate/Libra",
            "Libra_mtea_info.csv",
        ),
        index=False,
    )
    ParallelBackend().close(force=True)


def convert_PSM():
    train_df = pd.read_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/多元时序异常检测数据集汇总/PSM/train.csv")
    test_df = pd.read_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/多元时序异常检测数据集汇总/PSM/test.csv")
    test_label_df = pd.read_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/多元时序异常检测数据集汇总/PSM/test_label.csv")

    label_column = test_label_df.iloc[:, -1]

    # 将最后一列数据添加到 test_df 中，并设置列名为 "label"
    test_df["label"] = label_column

    train_df["label"] = 0
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = combined_df.iloc[:, 1:]
    # 将所有列堆叠在一起，data在前，cols在后
    melted_df = combined_df.melt(value_name="data", var_name="cols")

    # 调整列的顺序
    melted_df = melted_df[["data", "cols"]]
    cols_nums = melted_df["cols"].nunique()
    print(cols_nums)
    date_column = pd.Series(
        list(range(1, len(combined_df) + 1)) * cols_nums, dtype="int64"
    )
    melted_df.insert(0, "date", date_column)

    name = ['PSM' + ";" + 'PSM'] * len(combined_df) * cols_nums
    melted_df.insert(3, "name", name)

    melted_df.to_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/PSM.csv", index=False)
    print(melted_df)
    print(melted_df.shape)


def convert_MSL():
    train_df = pd.read_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/多元时序异常检测数据集汇总/MSL/MSL_train.csv")
    test_df = pd.read_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/多元时序异常检测数据集汇总/MSL/MSL_test.csv")

    # 将最后一列数据添加到 test_df 中，并设置列名为 "label"

    train_df["label"] = 0

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = combined_df.iloc[:, 1:]
    # 将所有列堆叠在一起，data在前，cols在后
    melted_df = combined_df.melt(value_name="data", var_name="cols")

    # 调整列的顺序
    melted_df = melted_df[["data", "cols"]]
    # melted_df.insert(0, 'date', range(1, len(melted_df)+1))
    cols_nums = melted_df["cols"].nunique()
    print(cols_nums)
    date_column = pd.Series(
        list(range(1, len(combined_df) + 1)) * cols_nums, dtype="int64"
    )
    melted_df.insert(0, "date", date_column)

    name = ['MSL' + ";" + 'MSL'] * len(combined_df) * cols_nums
    melted_df.insert(3, "name", name)

    melted_df.to_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/MSL.csv", index=False)
    print(melted_df)
    print(melted_df.shape)

def convert_SMAP():
    train_df = pd.read_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/多元时序异常检测数据集汇总/SMAP/SMAP_train.csv")
    test_df = pd.read_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/多元时序异常检测数据集汇总/SMAP/SMAP_test.csv")

    # 将最后一列数据添加到 test_df 中，并设置列名为 "label"

    train_df["label"] = 0

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = combined_df.iloc[:, 1:]
    # 将所有列堆叠在一起，data在前，cols在后
    melted_df = combined_df.melt(value_name="data", var_name="cols")

    # 调整列的顺序
    melted_df = melted_df[["data", "cols"]]
    # melted_df.insert(0, 'date', range(1, len(melted_df)+1))
    cols_nums = melted_df["cols"].nunique()
    print(cols_nums)
    date_column = pd.Series(
        list(range(1, len(combined_df) + 1)) * cols_nums, dtype="int64"
    )
    melted_df.insert(0, "date", date_column)

    name = ['SMAP' + ";" + 'SMAP'] * len(combined_df) * cols_nums
    melted_df.insert(3, "name", name)

    melted_df.to_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/SMAP.csv", index=False)
    print(melted_df)
    print(melted_df.shape)

def convert_SMD():
    for dir in os.listdir(r"D:\project\Datas\多元时序异常检测数据集汇总\SMD"):
        for file in os.listdir(
            os.path.join(r"D:\project\Datas\多元时序异常检测数据集汇总\SMD", dir)
        ):
            if file.endswith("_train.csv"):
                train_df = pd.read_csv(
                    os.path.join(r"D:\project\Datas\多元时序异常检测数据集汇总\SMD", dir, file)
                )
            if file.endswith("_test.csv"):
                test_df = pd.read_csv(
                    os.path.join(r"D:\project\Datas\多元时序异常检测数据集汇总\SMD", dir, file)
                )

        # 将最后一列数据添加到 test_df 中，并设置列名为 "label"

        train_df["label"] = 0

        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        combined_df = combined_df.iloc[:, 1:]
        # 将所有列堆叠在一起，data在前，cols在后
        melted_df = combined_df.melt(value_name="data", var_name="cols")

        # 调整列的顺序
        melted_df = melted_df[["data", "cols"]]
        # cols_nums = melted_df['cols'].unique().sum()
        # melted_df.insert(0, 'date', range(1, len(melted_df)+1))
        date_column = pd.Series(range(1, len(melted_df) + 1), dtype="int64")
        melted_df.insert(0, "date", date_column)

        file_name = dir + ".csv"
        melted_df.to_csv(
            os.path.join(r"C:\Users\86188\Desktop\单变量160\test", file_name), index=False
        )
        print(melted_df)
        print(melted_df.shape)


# convert_SMD()

# for file in os.listdir(r'C:\Users\86188\Desktop\单变量160\merge'):
#     file_path = os.path.join(r'C:\Users\86188\Desktop\单变量160\merge', file)
#     covert_tsb_uad_univarite_series(file_path, file)


def convert_another_SMD():
    train_df = pd.DataFrame(np.load(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/SMD/SMD_train.npy"))
    test_df = pd.DataFrame(np.load(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/SMD/SMD_test.npy"))
    test_label_df = pd.DataFrame(
        np.load(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/SMD/SMD_test_label.npy")
    )

    label_column = test_label_df.iloc[:, -1]

    # 将最后一列数据添加到 test_df 中，并设置列名为 "label"
    test_df["label"] = label_column

    train_df["label"] = 0
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    # 将所有列堆叠在一起，data在前，cols在后
    melted_df = combined_df.melt(value_name="data", var_name="cols")

    # 调整列的顺序
    melted_df = melted_df[["data", "cols"]]
    cols_nums = melted_df["cols"].nunique()
    print(cols_nums)
    date_column = pd.Series(
        list(range(1, len(train_df) + len(test_df) + 1)) * cols_nums, dtype="int64"
    )
    melted_df.insert(0, "date", date_column)
    name = ['SMD' + ";" + 'SMD'] * len(combined_df) * cols_nums
    melted_df.insert(3, "name", name)

    melted_df.to_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/SMD.csv", index=False)
    print(melted_df)
    print(melted_df.shape)


def convert_swat():
    train_df = pd.read_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/多元时序异常检测数据集汇总/swat/swat_train.csv")
    test_df = pd.read_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/多元时序异常检测数据集汇总/swat/swat_test.csv")

    # 将最后一列数据添加到 test_df 中，并设置列名为 "label"
    train_df.rename(columns={"Normal/Attack": "label"}, inplace=True)
    test_df.rename(columns={"Normal/Attack": "label"}, inplace=True)
    # train_df["label"] = 0

    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    # 将所有列堆叠在一起，data在前，cols在后
    melted_df = combined_df.melt(value_name="data", var_name="cols")

    # 调整列的顺序
    melted_df = melted_df[["data", "cols"]]
    # melted_df.insert(0, 'date', range(1, len(melted_df)+1))
    cols_nums = melted_df["cols"].nunique()
    print(cols_nums)
    date_column = pd.Series(
        list(range(1, len(combined_df) + 1)) * cols_nums, dtype="int64"
    )
    melted_df.insert(0, "date", date_column)

    name = ['swat' + ";" + 'swat'] * len(combined_df) * cols_nums
    melted_df.insert(3, "name", name)

    melted_df.to_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/ad_multi/swat.csv", index=False)
    print(melted_df)
    print(melted_df.shape)

def convert_ETT1():
    data = pd.read_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/forcast_multi/多元时序预测数据集汇总/csv格式/exchange_rate.csv")
    combined_df = data.iloc[:, 1:]
    # 将所有列堆叠在一起，data在前，cols在后
    melted_df = combined_df.melt(value_name="data", var_name="cols")

    # 调整列的顺序
    melted_df = melted_df[["data", "cols"]]
    cols_nums = melted_df["cols"].nunique()
    print(cols_nums)
    date_column = pd.Series(list(data.iloc[:, 0]) * cols_nums)

    melted_df.insert(0, "date", date_column)

    name = ['exchange_rate' + ";" + 'exchange_rate'] * len(combined_df) * cols_nums
    melted_df.insert(3, "name", name)

    melted_df.to_csv(
        r"/Users/xiangfeiqiu/D/datasets/processed_data/forcast_multi/exchange_rate.csv", index=False
    )
    print(melted_df)
    print(melted_df.shape)


# convert_ETT1()
def convert_pems03():
    data = pd.read_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/forcast_multi/多元时序预测数据集汇总/PEMS_data/PEMS03/PEMS03_data.csv")
    combined_df = data.iloc[:, :]
    # 将所有列堆叠在一起，data在前，cols在后
    melted_df = combined_df.melt(value_name="data", var_name="cols")

    # 调整列的顺序
    melted_df = melted_df[["data", "cols"]]
    cols_nums = melted_df["cols"].nunique()
    print(cols_nums)
    date_column = pd.Series(list(range(1, len(data) + 1)) * cols_nums, dtype="int64")

    melted_df.insert(0, "date", date_column)

    name = ['PEMS' + ";" + 'PEMS03'] * len(combined_df) * cols_nums
    melted_df.insert(3, "name", name)

    melted_df.to_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/forcast_multi/PEMS03.csv", index=False)
    print(melted_df)
    print(melted_df.shape)



def convert_pems03():
    data = pd.read_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/forcast_multi/多元时序预测数据集汇总/PEMS_data/PEMS03/PEMS03_data.csv")
    combined_df = data.iloc[:, :]
    # 将所有列堆叠在一起，data在前，cols在后
    melted_df = combined_df.melt(value_name="data", var_name="cols")

    # 调整列的顺序
    melted_df = melted_df[["data", "cols"]]
    cols_nums = melted_df["cols"].nunique()
    print(cols_nums)
    date_column = pd.Series(list(range(1, len(data) + 1)) * cols_nums, dtype="int64")

    melted_df.insert(0, "date", date_column)

    name = ['PEMS' + ";" + 'PEMS03'] * len(combined_df) * cols_nums
    melted_df.insert(3, "name", name)

    melted_df.to_csv(r"/Users/xiangfeiqiu/D/datasets/processed_data/forcast_multi/PEMS03.csv", index=False)
    print(melted_df)
    print(melted_df.shape)

import numpy as np



# 现在，three_dimensional_data 包含了你的三维数据，你可以使用它进行进一步的操作

# df = read_data(r'D:\project\self_pipeline\dataset\hospital_dataset.csv')
# print(df)
# is_datetime_index = isinstance(df.index, pd.DatetimeIndex)
# print(is_datetime_index)


# import os
#
# directory = "/Users/xiangfeiqiu/D/datasets/processed_data/ad_univariate/tsb_uad/origin_tsb_uad/NASA-SMAP"
# directory1 = "/Users/xiangfeiqiu/D/datasets/processed_data/ad_univariate/tsb_uad/origin_tsb_uad/NASA-SMAP-new"
#
# # 获取目录下所有文件
# files = os.listdir(directory)
#
# # 用于存储train.out文件长度的字典，以前缀为键
# train_lengths = {}
#
# # 遍历文件
# for file in files:
#     # 分割文件名为前缀和后缀
#     prefix, extension = file.split('.', 1)
#
#     # 如果后缀是train.out
#     if extension == 'train.out':
#         # 记录train.out文件长度
#         with open(os.path.join(directory, file), 'r') as f:
#             raw_data = pd.read_csv(os.path.join(directory, file), header=None)
#             series_len = len(raw_data)
#             train_data = f.read()
#             # 记录train.out文件长度
#             if prefix not in train_lengths:
#                 train_lengths[prefix] = 0
#             train_lengths[prefix] += series_len
#
#             # 合并train.out和test.out文件
#             test_file = f"{prefix}.test.out"
#             if test_file in files:
#                 with open(os.path.join(directory, test_file), 'r') as test_f:
#                     test_data = test_f.read()
#                     # 合并train.out和test.out内容
#                     merged_data = train_data + test_data
#
#                     # 写入新文件
#                     new_file_name = f"{prefix}_{train_lengths[prefix]}.csv"
#                     with open(os.path.join(directory1, new_file_name), 'w') as new_f:
#                         new_f.write(merged_data)
#                         print(f"Merged and saved to {new_file_name}")
#
# print("Train.out file lengths:")
# for prefix, length in train_lengths.items():
#     print(f"{prefix}: {length}")
