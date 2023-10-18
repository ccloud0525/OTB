# -*- coding: utf-8 -*-
class Singleton(type):
    """
    用于通过meta class的方法构造单例类
    """

    _instance_dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance_dict:
            cls._instance_dict[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance_dict[cls]
