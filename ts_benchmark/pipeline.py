# -*- coding: utf-8 -*-
from typing import List

from ts_benchmark.data_loader.data_loader import load_data
from ts_benchmark.evaluation.evaluate_model import eval_model
from ts_benchmark.models.get_model import get_model
from ts_benchmark.report.save_log import save_log


def pipeline(
    data_loader_config: dict, model_config: dict, model_eval_config: dict,
) -> List[str]:
    """
     执行benchmark的pipeline流程，包括加载数据、构建模型、评估模型并生成报告。

     :param data_loader_config: 数据加载的配置。
     :param model_config: 模型构建的配置。
     :param model_eval_config: 模型评估的配置。
     """

    # 加载数据
    series_list = load_data(data_loader_config)

    # 构建模型
    model_factory_list = get_model(model_config)

    # 循环遍历每个模型
    log_file_names = []
    for index, model_factory in enumerate(model_factory_list):
        # 评估模型
        for i, result_df in enumerate(eval_model(
            model_factory, series_list, model_eval_config
        )):

            # 获得测评的模型名称
            model_name = model_config["models"][index]["model_name"].split(".")[-1]

            # 生成报告
            log_file_names.append(save_log(
                result_df,
                model_name if i == 0 else f"{model_name}_{i}",
            ))

    return log_file_names
