from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler


class LOF:
    """
    LOF (Local Outlier Factor) 模型类，用于异常检测。

    LOF 是一种基于密度的异常检测方法，用于识别在数据集中与其邻居相比具有显著不同密度的数据点。
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2,
        metric_params: dict = None,
        contamination: float = 0.1,
        n_jobs: int = 1,
    ):
        """
        初始化 LOF 模型。

        :param n_neighbors: 用于计算 LOF 的邻居数量。
        :param algorithm: LOF 计算所使用的算法。
        :param leaf_size: 构造 KD 树或球树时使用的叶子大小。
        :param metric: 用于计算距离的距离度量。
        :param p: 距离度量中的参数 p。
        :param metric_params: 距离度量的其他参数。
        :param contamination: 预期异常样本的比例。
        :param n_jobs: 并行计算所使用的工作线程数。
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.model_name = "LOF"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        返回 LOF 模型所需的超参数。

        :return: 一个空字典，表示 LOF 模型不需要额外的超参数。
        """
        return {}

    def detect_fit(self, X, y=None):
        """
        训练 LOF 模型。

        :param X: 训练数据。
        :param y: 标签数据（可选）。
        """
        pass

    def detect_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用 LOF 模型计算异常得分。

        :param X: 待计算得分的数据。
        :return: 异常得分数组。
        """
        X = X.values.reshape(-1, 1)

        self.detector_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            contamination=self.contamination,
            n_jobs=self.n_jobs,
        )
        self.detector_.fit(X=X)

        self.decision_scores_ = -self.detector_.negative_outlier_factor_

        score = (
            MinMaxScaler(feature_range=(0, 1))
            .fit_transform(self.decision_scores_.reshape(-1, 1))
            .ravel()
        )
        return score

    def detect_label(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用 LOF 模型进行异常检测并生成标签。

        :param X: 待检测的数据。
        :return: 异常标签数组。
        """
        X = X.values.reshape(-1, 1)

        self.detector_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            contamination=self.contamination,
            n_jobs=self.n_jobs,
        )
        self.detector_.fit(X=X)

        self.decision_scores_ = -self.detector_.negative_outlier_factor_

        score = (
            MinMaxScaler(feature_range=(0, 1))
            .fit_transform(self.decision_scores_.reshape(-1, 1))
            .ravel()
        )
        return score

    def __repr__(self) -> str:
        """
        返回模型名称的字符串表示。
        """
        return self.model_name
