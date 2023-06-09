import numpy as np

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from typing import Any, List


def calc_validation_metric(window: Any, y_pred) -> List[np.ndarray]:
    """
    Calculate MAPE on all validation windows
    :param window: class window generator
    :param y_pred: predicted validation value
    :return list_metric: MAPE and RMSE on all validation window
    """
    X_train, y_train, X_val, y_val, X_test, y_test = window.get_data_to_model()
    list_num_wind = np.arange(X_val.shape[0])
    list_mape = []
    list_rmse = []
    for wind in list_num_wind:
        y_val_inv, pred_val_inv = window.get_inverse_values(
            y_val[wind, :, :], y_pred[wind, :, :]
        )
        mape = mean_absolute_percentage_error(y_val_inv[:, -1], pred_val_inv[:, -1])
        rmse = mean_squared_error(y_val_inv[:, -1], pred_val_inv[:, -1], squared=False)
        list_mape.append(mape * 100)
        list_rmse.append(rmse)
    return [np.array(list_mape), np.array(list_rmse)]


def calc_test_metric(window: Any, y_pred) -> List[np.ndarray]:
    """
    Calculate MAPE on all test windows
    :param window: class window generator
    :param y_pred: predicted test value
    :return list_metric: MAPE and RMSE on all test window
    """
    X_train, y_train, X_val, y_val, X_test, y_test = window.get_data_to_model()
    list_num_wind = np.arange(X_test.shape[0])
    list_mape = []
    list_rmse = []
    for wind in list_num_wind:
        y_test_inv, pred_val_inv = window.get_inverse_values(
            y_test[wind, :, :], y_pred[wind, :, :]
        )
        mape = mean_absolute_percentage_error(y_test_inv[:, -1], pred_val_inv[:, -1])
        rmse = mean_squared_error(y_test_inv[:, -1], pred_val_inv[:, -1], squared=False)
        list_mape.append(mape * 100)
        list_rmse.append(rmse)
    return [np.array(list_mape), np.array(list_rmse)]
