# -*- coding: utf-8 -*-
import importlib
from typing import Type, Any

from ts_benchmark.baselines import ADAPTER


def get_model_info(model_config: dict) -> Any:
    """
    根据模型配置获取模型信息。
    根据提供的模型配置信息，检索并返回相应的模型类。

    :param model_config: 包含模型配置信息的字典。
    :return: 与指定 model_name 相对应的模型类。
    :raises ImportError: 如果无法导入指定的模型包。
    :raises AttributeError: 如果在导入的模块中找不到指定的 class_name。
    """

    def import_model_info(model_class: str) -> Any:
        """
        导入模型信息。

        根据提供的模型类名，从对应的模块中导入并返回该模型类。

        :param model_class: 要导入的模型类的全限定名，例如 'package.module.ModelClassName'。

        :return: 导入的模型类。
        """
        model_package, class_name = model_class.rsplit(".", 1)
        # 导入指定的模型包
        mod = importlib.import_module(model_package)
        # 从导入的模块中检索并返回指定的类
        model_info = getattr(mod, class_name)
        return model_info

    model_name = model_config["model_name"]

    try:
        model_class = model_name
        model_info = import_model_info(model_class)

    except (ImportError, AttributeError):
        model_class = "ts_benchmark.baselines." + model_config["model_name"]
        model_info = import_model_info(model_class)

    adapter_name = model_config.get("adapter")
    if adapter_name is not None:
        if adapter_name not in ADAPTER:
            raise ValueError(f"Unknown adapter {adapter_name}")
        model_info = import_model_info(ADAPTER[adapter_name])(model_info)
        
    return model_info


def get_model_hyper_params(
    recommend_model_hyper_params: dict, required_hyper_params: dict, model_config: dict
) -> dict:
    """
    获取模型的超参数。

    根据推荐的模型超参数、所需的超参数映射以及模型配置，返回合并后的模型超参数。

    :param recommend_model_hyper_params: 推荐的模型超参数。
    :param required_hyper_params: 所需的超参数映射，格式为 {参数名: 标准参数名}。
    :param model_config: 模型配置，包括 model_hyper_params 字段。

    :return: 合并后的模型超参数。

    :raises ValueError: 如果有缺失的超参数。
    """
    model_hyper_params = {
        arg_name: recommend_model_hyper_params[arg_std_name]
        for arg_name, arg_std_name in required_hyper_params.items()
        if arg_std_name in recommend_model_hyper_params
    }
    model_hyper_params.update(model_config.get("model_hyper_params", {}))
    missing_hp = set(required_hyper_params) - set(model_hyper_params)
    if missing_hp:
        raise ValueError("These hyper parameters are missing : {}".format(missing_hp))
    return model_hyper_params


class ModelFactory:
    """
    模型工厂类，用于实例化模型。
    """

    def __init__(
        self, model_factory: Type, model_hyper_params: dict,
    ):
        """
        初始化 ModelFactory 对象。

        :param model_factory: 实际模型工厂类，用于创建模型实例。
        :param model_hyper_params: 模型所需的超参数字典，包含标准名称映射。
        """
        self.model_factory = model_factory
        self.model_hyper_params = model_hyper_params

    def __call__(self) -> Any:
        """
        通过调用实际模型工厂类来实例化模型。

        :return: 实例化的模型对象。
        """

        return self.model_factory(**self.model_hyper_params)

def get_model(all_model_config: dict) -> list:
    """
    根据模型配置获取模型工厂列表。
    根据提供的全部模型配置信息，创建模型工厂列表用于实例化模型。

    :param all_model_config: 包含所有模型配置信息的字典。
    :return: 模型工厂列表，用于实例化不同模型。
    """
    model_factory_list = []  # 存储模型工厂的列表
    # 遍历每个模型配置
    for model_config in all_model_config["models"]:
        model_info = get_model_info(model_config)  # 获取模型信息

        # 解析模型信息
        if isinstance(model_info, dict):
            model_factory = model_info.get("model_factory")
            if model_factory is None:
                raise ValueError("model_factory is none")
            required_hyper_params = model_info.get("required_hyper_params", {})

        elif isinstance(model_info, type):
            model_factory = model_info
            required_hyper_params = {}
            if hasattr(model_factory, "required_hyper_params"):
                required_hyper_params = model_factory.required_hyper_params()
        else:
            model_factory = model_info
            required_hyper_params = {}

        model_hyper_params = get_model_hyper_params(
            all_model_config["recommend_model_hyper_params"],
            required_hyper_params,
            model_config,
        )
        # 添加模型工厂到列表
        model_factory_list.append(ModelFactory(model_factory, model_hyper_params))
    return model_factory_list