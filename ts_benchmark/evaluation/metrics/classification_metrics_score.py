# -*- coding: utf-8 -*-

import numpy as np
from sklearn import metrics
from ts_benchmark.evaluation.metrics.vus_metrics import metricor, generate_curve
from ts_benchmark.evaluation.metrics.utils import get_list_anomaly, find_length

__all__ = ["auc_roc", "auc_pr", "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR"]


metricor_grader = metricor()


def auc_roc(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return metrics.roc_auc_score(actual, predicted)


def auc_pr(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return metrics.average_precision_score(actual, predicted)


def R_AUC_ROC(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    slidingWindow = 100
    R_AUC_ROC, R_AUC_PR, _, _, _ = metricor_grader.RangeAUC(
        labels=actual, score=predicted, window=slidingWindow, plot_ROC=True
    )
    return R_AUC_ROC


def R_AUC_PR(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    slidingWindow = 100
    R_AUC_ROC, R_AUC_PR, _, _, _ = metricor_grader.RangeAUC(
        labels=actual, score=predicted, window=slidingWindow, plot_ROC=True
    )
    return R_AUC_PR


def VUS_ROC(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    slidingWindow = 100

    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
        actual, predicted, 2 * slidingWindow
    )
    return VUS_ROC


def VUS_PR(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    slidingWindow = int(np.median(get_list_anomaly(actual)))
    slidingWindow = 100

    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
        actual, predicted, 2 * slidingWindow
    )
    return VUS_PR


# y_test = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
# pred_labels = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1])
# a = VUS_ROC(y_test, pred_labels)
# b = VUS_PR(y_test, pred_labels)
# # vus_results = get_range_vus_roc(y_test, pred_labels, 100)  # default slidingWindow = 100
# print("VUS_ROC", a)
# print("VUS_PR", b)

#
# import numpy as np
#
#
# score = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# label = np.array([0, 1, 1, 0, 1, 1, 1, 1])
# print("auc_roc:", auc_roc(label, score, label))
# print("auc_pr:", auc_pr(label, score, label))
# print("R_AUC_ROC:", R_AUC_ROC(label, score, label))
# print("R_AUC_PR:", R_AUC_PR(label, score, label))
#
# print("VUS_ROC:", VUS_ROC(label, score, label))
# print("VUS_PR:", VUS_PR(label, score, label))
