import os.path
from datetime import datetime
from distutils.util import strtobool

import numpy as np
import pandas as pd

# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the data_loader: frequency, horizon, whether the data_loader contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
from ts_benchmark.utils.data_processing import read_data


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
def convert_monash_multivariate(path):
    (
        loaded_data,
        frequency,
        forecast_horizon,
        contain_missing_values,
        contain_equal_length,
    ) = convert_tsf_to_dataframe(path)
    frequency_dict = {}
    frequency_dict["weekly"] = "W"
    frequency_dict["daily"] = "D"
    frequency_dict["yearly"] = "Y"
    frequency_dict["monthly"] = "MS"

    df = pd.DataFrame()
    start_time = loaded_data.iloc[0, 0]
    n_cols = loaded_data.iloc[:, 0].shape[0]
    time_points = loaded_data.iloc[0, 1].shape[0]
    timestamp = pd.date_range(
        start_time, freq=frequency_dict[frequency], periods=time_points
    )
    df.insert(0, "date", timestamp)
    time_col = df
    data_list = []
    cols_list = []

    for i in range(n_cols):
        if i != n_cols - 1:
            time_col = pd.concat([time_col, df], axis=0)
        data_list = data_list + loaded_data.iloc[i, 1].tolist()
        col_mame = "col" + "_" + str(i + 1)
        col = [col_mame] * time_points
        cols_list = cols_list + col

    df_new = pd.DataFrame()
    df_new.insert(0, "date", time_col)
    df_new.insert(1, "data", data_list)
    df_new.insert(2, "cols", cols_list)

    root_path = r"C:\Users\86188\Desktop\Monash_data_file_change"
    # root_path = r"D:\project\self_pipeline\dataset"
    file_name = path.split("\\")[-1].split(".")[0]
    new_file_name = file_name + ".csv"
    path = os.path.join(root_path, new_file_name)
    df_new.to_csv(path, index=False)

# Monash单变量数据集转换成our 格式
def convert_monash_univariate(path):
    (
        loaded_data,
        frequency,
        forecast_horizon,
        contain_missing_values,
        contain_equal_length,
    ) = convert_tsf_to_dataframe(path)
    frequency_dict = {
        'daily': 'days',
        'weekly': 'weeks',
        'monthly': 'months',
        'yearly': 'years',
        'hourly': 'hours'
    }
    for i in range(loaded_data.shape[0]):
        df = pd.DataFrame()
        start_time = loaded_data.iloc[i, 0]
        num = loaded_data.iloc[i, 1].shape[0]
        date_range = [(start_time + pd.DateOffset(**{frequency_dict[frequency]: i})).strftime("%Y-%m-%d %H:%M:%S") for i
                      in range(num)]

        cols = ["col_1"] * num
        df.insert(0, "date", date_range)
        df.insert(1, "data", loaded_data.iloc[i, 1])
        df.insert(2, "cols", cols)
        root_path = r"C:\Users\86188\Desktop\Monash_data_file_change"
        file_name = path.split("\\")[-1].split(".")[0]
        new_file_name = file_name + "_" + str(i + 1) + ".csv"
        file_path = os.path.join(root_path, new_file_name)
        df.to_csv(file_path, index=False)


# convert_monash_univariate(r"D:\project\Datas\单变量时间序列预测\Monash\原始数据\m4_daily_dataset.tsf")


def covert_tsb_uad_univarite_series(path):
    raw_data = pd.read_csv(path, header=None)
    series_len = len(raw_data)
    data = pd.concat([raw_data.iloc[:, 0], raw_data.iloc[:, 1]], ignore_index=True)
    col_name = ['col1'] * series_len + ['label'] * series_len
    new_df = pd.DataFrame({'data': data, 'cols': col_name})
    # 获取文件名（不包含路径）
    file_name = os.path.basename(path)
    # 修改文件名的扩展名为 ".csv"
    new_file_name = os.path.splitext(file_name)[0] + ".csv"
    file_path = os.path.join(r"C:\Users\86188\Desktop", new_file_name)
    new_df.to_csv(file_path, index=False)
    print(new_df)

# covert_tsb_uad_univarite_series(r"D:\project\self_pipeline\dataset\S01R02E0.baseline_result.csv@6.out")

def convert_PSM():
    train_df = pd.read_csv(r"D:\project\Datas\多元时序异常检测数据集汇总\PSM\train.csv")
    test_df = pd.read_csv(r"D:\project\Datas\多元时序异常检测数据集汇总\PSM\test.csv")
    test_label_df = pd.read_csv(r"D:\project\Datas\多元时序异常检测数据集汇总\PSM\test_label.csv")

    label_column = test_label_df.iloc[:, -1]

    # 将最后一列数据添加到 test_df 中，并设置列名为 "label"
    test_df['label'] = label_column

    train_df['label'] = 0
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = combined_df.iloc[:,1:]
    # 将所有列堆叠在一起，data在前，cols在后
    melted_df = combined_df.melt(value_name='data', var_name='cols')

    # 调整列的顺序
    melted_df = melted_df[['data', 'cols']]
    cols_nums = melted_df['cols'].nunique()
    print(cols_nums)
    date_column = pd.Series(list(range(1, len(combined_df) + 1)) * cols_nums, dtype='int64')
    melted_df.insert(0, 'date', date_column)
    melted_df.to_csv(r'D:\project\self_pipeline\dataset\PSM2.csv', index=False)
    print(melted_df)
    print(melted_df.shape)

def convert_MSL():
    train_df = pd.read_csv(r"D:\project\self_pipeline\dataset\MSL_train.csv")
    test_df = pd.read_csv(r"D:\project\self_pipeline\dataset\MSL_test.csv")


    # 将最后一列数据添加到 test_df 中，并设置列名为 "label"

    train_df['label'] = 0

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = combined_df.iloc[:,1:]
    # 将所有列堆叠在一起，data在前，cols在后
    melted_df = combined_df.melt(value_name='data', var_name='cols')

    # 调整列的顺序
    melted_df = melted_df[['data', 'cols']]
    # melted_df.insert(0, 'date', range(1, len(melted_df)+1))
    cols_nums = melted_df['cols'].nunique()
    print(cols_nums)
    date_column = pd.Series(list(range(1, len(combined_df) + 1)) * cols_nums, dtype='int64')
    melted_df.insert(0, 'date', date_column)

    melted_df.to_csv(r'D:\project\self_pipeline\dataset\MSL1.csv', index=False)
    print(melted_df)
    print(melted_df.shape)

def convert_SMAP():
    train_df = pd.read_csv(r"D:\project\Datas\多元时序异常检测数据集汇总\SMAP\SMAP_train.csv")
    test_df = pd.read_csv(r"D:\project\Datas\多元时序异常检测数据集汇总\SMAP\SMAP_test.csv")


    # 将最后一列数据添加到 test_df 中，并设置列名为 "label"

    train_df['label'] = 0

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = combined_df.iloc[:,1:]
    # 将所有列堆叠在一起，data在前，cols在后
    melted_df = combined_df.melt(value_name='data', var_name='cols')

    # 调整列的顺序
    melted_df = melted_df[['data', 'cols']]
    # melted_df.insert(0, 'date', range(1, len(melted_df)+1))
    cols_nums = melted_df['cols'].nunique()
    print(cols_nums)
    date_column = pd.Series(list(range(1, len(combined_df) + 1)) * cols_nums, dtype='int64')
    melted_df.insert(0, 'date', date_column)

    melted_df.to_csv(r'D:\project\self_pipeline\dataset\SMAP1.csv', index=False)
    print(melted_df)
    print(melted_df.shape)

def convert_SMD():
    for dir in os.listdir(r'D:\project\Datas\多元时序异常检测数据集汇总\SMD'):
        for file in os.listdir(os.path.join(r'D:\project\Datas\多元时序异常检测数据集汇总\SMD', dir)):
            if file.endswith('_train.csv'):
                train_df = pd.read_csv(os.path.join(r'D:\project\Datas\多元时序异常检测数据集汇总\SMD', dir, file))
            if file.endswith('_test.csv'):
                test_df = pd.read_csv(os.path.join(r'D:\project\Datas\多元时序异常检测数据集汇总\SMD', dir, file))

        # 将最后一列数据添加到 test_df 中，并设置列名为 "label"

        train_df['label'] = 0

        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        combined_df = combined_df.iloc[:,1:]
        # 将所有列堆叠在一起，data在前，cols在后
        melted_df = combined_df.melt(value_name='data', var_name='cols')

        # 调整列的顺序
        melted_df = melted_df[['data', 'cols']]
        # cols_nums = melted_df['cols'].unique().sum()
        # melted_df.insert(0, 'date', range(1, len(melted_df)+1))
        date_column = pd.Series(range(1, len(melted_df) + 1), dtype='int64')
        melted_df.insert(0, 'date', date_column)

        file_name = dir + '.csv'
        melted_df.to_csv(os.path.join(r'C:\Users\86188\Desktop\单变量160\test', file_name), index=False)
        print(melted_df)
        print(melted_df.shape)
# convert_SMD()


def covert_tsb_uad_univarite_series(path, file):
    raw_data = pd.read_csv(path, header=0)
    series_len = len(raw_data)
    data = pd.concat([raw_data.iloc[:, 0], raw_data.iloc[:, 1]], ignore_index=True)
    col_name = ['col1'] * series_len + ['label'] * series_len
    new_df = pd.DataFrame({'data': data, 'cols': col_name})
    date_column = pd.Series(list(range(1, series_len + 1)) * 2, dtype='int64')
    new_df.insert(0, 'date', date_column)
    file_path = os.path.join(r"C:\Users\86188\Desktop\单变量160\convert", file)
    new_df.to_csv(file_path, index=False)


# for file in os.listdir(r'C:\Users\86188\Desktop\单变量160\merge'):
#     file_path = os.path.join(r'C:\Users\86188\Desktop\单变量160\merge', file)
#     covert_tsb_uad_univarite_series(file_path, file)

def convert_another_SMD():
    train_df = pd.DataFrame(np.load(r"D:\大学学习\下载的东东\Edge浏览器下载\SMD\SMD\SMD_train.npy"))
    test_df = pd.DataFrame(np.load(r"D:\大学学习\下载的东东\Edge浏览器下载\SMD\SMD\SMD_test.npy"))
    test_label_df = pd.DataFrame(np.load(r"D:\大学学习\下载的东东\Edge浏览器下载\SMD\SMD\SMD_test_label.npy"))

    label_column = test_label_df.iloc[:, -1]

    # 将最后一列数据添加到 test_df 中，并设置列名为 "label"
    test_df['label'] = label_column

    train_df['label'] = 0
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    # 将所有列堆叠在一起，data在前，cols在后
    melted_df = combined_df.melt(value_name='data', var_name='cols')

    # 调整列的顺序
    melted_df = melted_df[['data', 'cols']]
    cols_nums = melted_df['cols'].nunique()
    print(cols_nums)
    date_column = pd.Series(list(range(1, len(train_df)+len(test_df) + 1)) * cols_nums, dtype='int64')
    melted_df.insert(0, 'date', date_column)
    melted_df.to_csv(r'D:\project\self_pipeline\dataset\SMD.csv', index=False)
    print(melted_df)
    print(melted_df.shape)



def convert_ETT1():
    data = pd.read_csv(r"D:\project\Datas\多元时序预测数据集汇总\csv格式\hospital_dataset.csv")
    combined_df = data.iloc[:,1:]
    # 将所有列堆叠在一起，data在前，cols在后
    melted_df = combined_df.melt(value_name='data', var_name='cols')

    # 调整列的顺序
    melted_df = melted_df[['data', 'cols']]
    cols_nums = melted_df['cols'].nunique()
    print(cols_nums)
    date_column = pd.Series(list(data.iloc[:, 0]) * cols_nums)

    melted_df.insert(0, 'date', date_column)

    melted_df.to_csv(r'D:\project\self_pipeline\dataset\hospital_dataset.csv', index=False)
    print(melted_df)
    print(melted_df.shape)

# convert_ETT1()
def convert_ETT():
    data = pd.read_csv(r"D:\project\Datas\多元时序预测数据集汇总\PEMS_data\PEMS03\PEMS03_data.csv")
    combined_df = data.iloc[:, :]
    # 将所有列堆叠在一起，data在前，cols在后
    melted_df = combined_df.melt(value_name='data', var_name='cols')

    # 调整列的顺序
    melted_df = melted_df[['data', 'cols']]
    cols_nums = melted_df['cols'].nunique()
    print(cols_nums)
    date_column = pd.Series(list(range(1, len(data)+ 1)) * cols_nums, dtype='int64')

    melted_df.insert(0, 'date', date_column)

    melted_df.to_csv(r'D:\project\self_pipeline\dataset\PEMS03_data.csv', index=False)
    print(melted_df)
    print(melted_df.shape)
# convert_ETT()

# df = read_data(r'D:\project\self_pipeline\dataset\hospital_dataset.csv')
# print(df)
# is_datetime_index = isinstance(df.index, pd.DatetimeIndex)
# print(is_datetime_index)

