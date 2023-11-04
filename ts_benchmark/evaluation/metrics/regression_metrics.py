# -*- coding: utf-8 -*-

import numpy as np

__all__ = ["mae", "mse", "rmse", "mape", "smape", "mase", 'wape', 'msmape']


def _error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Simple error """
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Percentage error """
    return (actual - predicted) / actual


def mse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """ Mean Absolute Error """

    return np.mean(np.abs(_error(actual, predicted)))


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    hist_data: np.ndarray,
    seasonality: int = 1,
    **kwargs
):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    scale = len(predicted) / (len(hist_data) - seasonality)

    dif = 0
    for i in range((seasonality + 1), len(hist_data)):
        dif = dif + abs(hist_data[i] - hist_data[i - seasonality])

    scale = scale * dif

    return (sum(abs(actual - predicted)) / scale)[0]


def mape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    """
    return np.mean(np.abs(_percentage_error(actual, predicted))) * 100


def smape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """
    Symmetric Mean Absolute Percentage Error
    """
    return (
        np.mean(
            2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)))
        )
        * 100
    )

def wape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    """Masked weighted absolute percentage error (WAPE)

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
    Returns:
        torch.Tensor: masked mean absolute error
    """
    loss = np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    return loss

def msmape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 0.1, **kwargs):
    """
    Function to calculate series wise smape values

    Parameters
    forecasts - a matrix containing forecasts for a set of series
                no: of rows should be equal to number of series and no: of columns should be equal to the forecast horizon
    test_set - a matrix with the same dimensions as 'forecasts' containing the actual values corresponding with them
    """

    comparator = np.full_like(actual, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted) + np.abs(actual) + epsilon)
    msmape_per_series = np.mean(2 * np.abs(predicted - actual) / denom) * 100
    return msmape_per_series

