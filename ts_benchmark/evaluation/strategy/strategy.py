# -*- coding: utf-8 -*-
import abc
import json
import logging


class Strategy(metaclass=abc.ABCMeta):
    """
    策略基类，用于定义时间序列预测策略的通用结构。
    """

    REQUIRED_FIELDS = []
    STRATEGY_NAME = "strategy_name"

    def __init__(self, model_eval_config: dict):
        """
        初始化策略对象。

        :param model_eval_config: 模型评估配置。
        """
        self.model_eval_config = model_eval_config

    @abc.abstractmethod
    def execute(self, series_name, model, evaluator):
        """
        执行策略的具体预测过程。

        """
        pass

    def get_config_str(self):
        """
        获取配置信息的字符串表示。

        :return: 配置信息的 JSON 格式字符串。
        """
        strategy_args = self.model_eval_config["strategy_args"]

        provided_args = sorted(list(strategy_args.keys()))
        required_args = ["strategy_name"] + self.REQUIRED_FIELDS

        if provided_args != sorted(required_args):
            missing_args = [
                arg for arg in self.REQUIRED_FIELDS if arg not in provided_args
            ]
            extra_args = [arg for arg in provided_args if arg not in required_args]
            config_args = {
                arg: strategy_args[arg] for arg in provided_args if arg in required_args
            }

            if missing_args:
                error_message = f"缺少参数: {', '.join(missing_args)} "
                raise RuntimeError(error_message)
            if extra_args:
                error_message = f"多出参数: {', '.join(extra_args)} "
                logging.warning(error_message)

            return json.dumps(config_args, sort_keys=True)
        else:
            return json.dumps(strategy_args, sort_keys=True)
