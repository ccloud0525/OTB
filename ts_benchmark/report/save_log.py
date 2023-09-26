# -*- coding: utf-8 -*-
import io
import json
import os

import pandas as pd

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.utils.compress import compress, get_compress_file_ext
from ts_benchmark.utils.get_file_name import get_log_file_name


def save_log(
    result_df: pd.DataFrame,
    model_name: str,
    compress_method: str = "gz",
) -> str:
    """
    保存日志数据。

    将评估结果、模型超参数、模型评估配置和模型名称保存到日志文件中。

    :param result_df: 评估结果的 DataFrame。
    :param model_name: 模型名称。
    :param compress_method: 输出文件压缩方式。
    """
    # model_hyper_params_str = json.dumps(model_hyper_params, sort_keys=True)

    result_path = os.path.join(ROOT_PATH, "result")
    os.makedirs(result_path, exist_ok=True)

    log_filename = get_log_file_name()
    file_path = os.path.join(result_path, model_name + log_filename)

    buf = io.StringIO()
    result_df.to_csv(buf, index=False)
    if compress_method is not None:
        write_data = compress({os.path.basename(file_path): buf.getvalue()}, method=compress_method)
        file_path = f"{file_path}.{get_compress_file_ext(compress_method)}"
    else:
        write_data = buf.getvalue().encode("utf8")

    with open(file_path, "wb") as fh:
        fh.write(write_data)

    return file_path
