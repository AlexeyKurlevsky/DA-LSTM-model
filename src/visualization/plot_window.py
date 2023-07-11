import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error
from matplotlib import dates
from typing import Any


def plot_validation_window(window: Any, y_pred: Any, save_plot_path: str) -> Any:
    """
    Plot 4 validation window
    :param window: class window generator
    :param y_pred: predicted validation value
    :param save_plot_path: path to save figure
    :return: figure
    """
    fig, ax = plt.subplots(2, 2, figsize=(20, 15))
    ax = ax.ravel()
    window_list = [0, 20, 40, 60]
    df_scaled = window.get_standard_data()
    X_train, y_train, X_val, y_val, X_test, y_test = window.get_data_to_model()
    df_val = df_scaled.iloc[
        window.conf.train_split - window.conf.window_size : window.conf.val_split, :
    ]
    date_val = df_val.index.values
    for ind, val in enumerate(window_list):
        assert X_val.shape[0] - 1 >= val, f"Окна c номером {val} не существует"
        y_val_inv, pred_val_inv = window.get_inverse_values(
            y_val[val, :, :], y_pred[val, :, :]
        )
        y_based_pred = window.scaler.inverse_transform(X_val[val, :, :])
        ax[ind].plot(
            date_val[val : val + window.conf.window_size],
            y_based_pred[:, -1],
            color="cornflowerblue",
        )
        ax[ind].plot(
            date_val[
                val
                + window.conf.window_size : val
                + window.conf.window_size
                + window.conf.n_future
            ],
            y_val_inv[:, -1],
            label="Факт",
            color="cornflowerblue",
        )
        ax[ind].plot(
            date_val[
                val
                + window.conf.window_size : val
                + window.conf.window_size
                + window.conf.n_future
            ],
            pred_val_inv[:, -1],
            label="Прогноз",
            linestyle="--",
            color="salmon",
        )
        ax[ind].axvline(
            date_val[val + window.conf.window_size], color="k", linestyle="--"
        )
        ax[ind].axvspan(
            date_val[val + window.conf.window_size],
            date_val[val + window.conf.window_size + window.conf.n_future],
            alpha=0.5,
            color="lightgray",
        )
        mape = (
            mean_absolute_percentage_error(y_val_inv[:, -1], pred_val_inv[:, -1]) * 100
        )
        ax[ind].set_title(f"Предсказание модели. МАРЕ={round(mape, 2)} %")
        ax[ind].xaxis.set_major_formatter(dates.DateFormatter("%d-%m-%y"))
        ax[ind].tick_params(axis="x", labelrotation=45)
        ax[ind].grid()
        ax[ind].legend()
    plt.savefig(save_plot_path)
    return fig


def plot_test_window(window: Any, y_pred: Any, save_plot_path: str) -> Any:
    """
    Plot 2 test window
    :param window: class window generator
    :param y_pred: predicted test value
    :param save_plot_path: path to save figure
    :return: figure
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax = ax.ravel()
    window_list = [0, 20]
    df_scaled = window.get_standard_data()
    X_train, y_train, X_val, y_val, X_test, y_test = window.get_data_to_model()
    df_test = df_scaled.iloc[window.conf.val_split - window.conf.window_size :, :]
    date_test = df_test.index.values
    for ind, val in enumerate(window_list):
        y_test_inv, pred_test_inv = window.get_inverse_values(
            y_test[val, :, :], y_pred[val, :, :]
        )
        y_based_pred = window.scaler.inverse_transform(X_test[val, :, :])
        ax[ind].plot(
            date_test[val : val + window.conf.window_size],
            y_based_pred[:, -1],
            color="cornflowerblue",
        )
        ax[ind].plot(
            date_test[
                val
                + window.conf.window_size : val
                + window.conf.window_size
                + window.conf.n_future
            ],
            y_test_inv[:, -1],
            label="Факт",
            color="cornflowerblue",
        )
        ax[ind].plot(
            date_test[
                val
                + window.conf.window_size : val
                + window.conf.window_size
                + window.conf.n_future
            ],
            pred_test_inv[:, -1],
            label="Прогноз",
            linestyle="--",
            color="salmon",
        )
        ax[ind].axvline(
            date_test[val + window.conf.window_size], color="k", linestyle="--"
        )
        ax[ind].axvspan(
            date_test[val + window.conf.window_size],
            date_test[val + window.conf.window_size + window.conf.n_future],
            alpha=0.5,
            color="lightgray",
        )
        mape = (
            mean_absolute_percentage_error(y_test_inv[:, -1], pred_test_inv[:, -1])
            * 100
        )
        ax[ind].set_title(f"Предсказание модели. МАРЕ={round(mape, 2)}.")
        ax[ind].xaxis.set_major_formatter(dates.DateFormatter("%d-%m-%y"))
        ax[ind].tick_params(axis="x", labelrotation=45)
        ax[ind].grid()
        ax[ind].legend()
    plt.savefig(save_plot_path)
    return fig
