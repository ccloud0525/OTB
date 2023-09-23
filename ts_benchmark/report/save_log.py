# -*- coding: utf-8 -*-
import json
import os

import pandas as pd

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.utils.get_file_name import get_log_file_name


def save_log(
    result_df: pd.DataFrame,
    model_hyper_params: dict,
    model_eval_config_str: str,
    model_name: str,
) -> str:
    """
    保存日志数据。

    将评估结果、模型超参数、模型评估配置和模型名称保存到日志文件中。

    :param result_df: 评估结果的 DataFrame。
    :param model_hyper_params: 模型超参数。
    :param model_eval_config_str: 模型评估配置的字符串表示。
    :param model_name: 模型名称。

    :return: None
    """
    model_hyper_params_str = json.dumps(model_hyper_params, sort_keys=True)

    result_path = os.path.join(ROOT_PATH, "result")
    os.makedirs(result_path, exist_ok=True)

    log_filename = get_log_file_name()
    file_path = os.path.join(result_path, model_name + log_filename)

    result_df_copy = result_df.copy()
    result_df_copy.insert(0, "model_params", model_hyper_params_str)
    result_df_copy.insert(0, "model_eval_config", model_eval_config_str)
    result_df_copy.insert(0, "model_name", model_name)

    result_df_copy.to_csv(file_path, index=False)  # 第一次写入时设置header=True

    return file_path
