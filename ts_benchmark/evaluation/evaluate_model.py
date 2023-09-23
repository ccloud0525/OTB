# -*- coding: utf-8 -*-
import logging
import traceback
from typing import Callable, Tuple, Any

import pandas as pd
import tqdm
from pandas import DataFrame

from ts_benchmark.data_loader.data_pool import DataPool
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.strategy import STRATEGY
from ts_benchmark.utils.parallel import ParallelBackend

logger = logging.getLogger(__name__)


def _safe_execute(fn, args, evaluator):
    """
    make sure execution does not crash even if there are exceptions
    """
    try:
        return fn(*args)
    except Exception as e:
        log = traceback.format_exc()
        return evaluator.default_result()[:-1] + [f"{log}\n{e}"]


def eval_model(
    model_factory: Callable, series_list: list, model_eval_config: dict
) -> Tuple[DataFrame, Any]:
    """
    评估模型在时间序列数据上的性能。
    根据提供的模型工厂、时间序列列表和评估配置，对模型进行评估，并返回评估结果的 DataFrame。

    :param model_factory: 模型工厂对象，用于创建模型实例。
    :param series_list: 包含时间序列名称的列表。
    :param model_eval_config: 评估配置信息，包括策略、评价指标等。
    :return: 包含评估结果的 DataFrame。
    """
    # 获取数据池实例，加载数据
    DataPool().prepare_data(series_list)

    # 获取策略类
    strategy_class = STRATEGY.get(model_eval_config["strategy_args"]["strategy_name"])
    if strategy_class is None:
        raise RuntimeError("strategy_class is none")

    # 解析评价指标配置
    metric = model_eval_config["metric_name"]
    if metric == "all":
        metric = list(strategy_class.accepted_metrics())
    elif isinstance(metric, (str, dict)):
        metric = [metric]

    metric = [
        {"name": metric_info} if isinstance(metric_info, str) else metric_info
        for metric_info in metric
    ]

    # 检查评价指标是否合法
    invalid_metrics = [
        m.get("name")
        for m in metric
        if m.get("name") not in strategy_class.accepted_metrics()
    ]
    if invalid_metrics:
        raise RuntimeError("要评测的评价指标不存在: {}".format(invalid_metrics))

    # 创建评估器实例
    evaluator = Evaluator(metric)

    strategy = strategy_class(model_eval_config)  # 创建评估策略对象

    eval_backend = ParallelBackend()

    result_list = []
    for series_name in tqdm.tqdm(series_list, desc="scheduling..."):
        model = model_factory()  # 创建模型实例
        # single_series_results = strategy.execute(
        #     series_name, model, evaluator
        # )  # 执行策略评估
        # TODO: refactor data model to optimize communication cost in parallel mode
        result_list.append(eval_backend.schedule(strategy.execute, (series_name, model, evaluator)))
        # result_list.append(single_series_results)

    result_list = [_safe_execute(it.result, (), evaluator) for it in tqdm.tqdm(result_list, desc="collecting...")]

    # 使用列表解析选择需要的列，并构造 DataFrame

    result_df = pd.DataFrame(result_list, columns=evaluator.metric_names)
    result_df.insert(0, "file_name", series_list)
    return result_df, strategy.get_config_str()
