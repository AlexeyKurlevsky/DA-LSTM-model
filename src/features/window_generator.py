import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import dates
from sklearn.metrics import mean_absolute_percentage_error
from typing import Any, Tuple, List
from sklearn.preprocessing import StandardScaler


class WindowGenerator:
    def __init__(
            self,
            data: pd.DataFrame,
            conf: Any,
            mean_flg: bool = False,
            scaler: Any = StandardScaler(),
    ):
        """
        Class that generates time series windows.
        This class implements data preparation, calculation of prediction metrics
        and visualization of the prediction result.
        :param data: raw dataframe
        :param conf: dictionary with research settings
        :param mean_flg: flag for using moving average in data preparation
        :param scaler: scaler from sklearn for normalization data
        """
        self.data = data
        self.conf = conf
        self.mean_flg = mean_flg
        self.scaler = scaler

    def get_standard_data(self) -> pd.DataFrame:
        """
        Data normalization
        """
        assert (
                self.data.columns[-1] == "Характерстика ДБ"
        ), "Неправильная последовательность столбцов"
        data_train = self.data.iloc[: self.conf.train_split, :]
        data_test = self.data.iloc[self.conf.train_split:, :]
        col_name = self.data.columns
        df_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(data_train),
            columns=col_name,
            index=data_train.index,
        )
        df_test_scaled = pd.DataFrame(
            self.scaler.transform(data_test), columns=col_name, index=data_test.index
        )
        if self.mean_flg:
            df_mean = (
                df_train_scaled.iloc[: self.conf.train_split, :]
                .rolling(window=self.conf.window_size_MA)
                .mean()
            )
            df_mean.dropna(inplace=True)
            df_scaled = pd.concat([df_mean, df_test_scaled])
        self.conf.n_samples = df_scaled.shape[0]
        assert (
                self.conf.val_split + self.conf.n_future <= self.conf.n_samples
        ), f"Некорректное разбиение"
        return df_scaled

    def plot_standard_data(self) -> None:
        """
        Plot prepared data for model training
        """
        df_scaled = self.get_standard_data()
        ax = df_scaled.plot(subplots=True)
        for ax_i in ax:
            ax_i.axvline(
                df_scaled.index[self.conf.train_split], color="k", linestyle="--"
            )
            ax_i.axvline(
                df_scaled.index[self.conf.val_split], color="b", linestyle="--"
            )
        plt.show()

    def _split_series(self, series: np.ndarray) -> List[np.ndarray]:
        """
        Splitting the time series into windows
        :param series: array with time series
        """
        X, y = list(), list()
        for window_start in range(len(series)):
            past_end = window_start + self.conf.window_size
            future_end = past_end + self.conf.n_future
            if future_end > len(series):
                break
            past, future = (
                series[window_start:past_end, :],
                series[past_end:future_end, :],
            )
            X.append(past)
            y.append(future)
        return [np.array(X), np.array(y)]

    def get_data_to_model(self) -> List[np.ndarray]:
        """
        Get train, validation and test windows
        """
        df_scaled = self.get_standard_data()

        X_train, y_train = self._split_series(
            df_scaled.iloc[: self.conf.train_split, :].values
        )
        X_val, y_val = self._split_series(
            df_scaled.iloc[
            self.conf.train_split - self.conf.window_size: self.conf.val_split, :
            ].values
        )
        X_test, y_test = self._split_series(
            df_scaled.iloc[self.conf.val_split - self.conf.window_size:, :].values
        )

        return [X_train, y_train, X_val, y_val, X_test, y_test]

    def get_tensor_data(self) -> List[tf.Tensor]:
        """
        Convert numpy array to Tensor. In the training set, the windows are mixed and divided into batches.
        In the validation set, they are only divided into batches.
        """
        X_train, y_train, X_val, y_val, X_test, y_test = self.get_data_to_model()
        BUFFER_SIZE = X_train.shape[0]

        train_data_multi = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_data_multi = (
            train_data_multi.cache()
            .shuffle(BUFFER_SIZE, seed=0)
            .batch(self.conf.batch_size)
            .repeat()
        )

        val_data_multi = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_data_multi = val_data_multi.batch(self.conf.batch_size).repeat()

        return [train_data_multi, val_data_multi]

    def get_inverse_values(self, y_true: Any, y_pred: Any) -> List[np.ndarray]:
        """
        Scale back the data to the original representation.
        :param y_true: fact value
        :param y_pred: predict value
        :return:
        """
        y_true_inv = self.scaler.inverse_transform(y_true)
        y_pred_inv = self.scaler.inverse_transform(y_pred)
        return [y_true_inv, y_pred_inv]

    def plot_validation_window(self, model: Any) -> None:
        """
        Plot 4 validation window
        :param model: RNN model
        :return:
        """
        fig, ax = plt.subplots(2, 2, figsize=(20, 15))
        ax = ax.ravel()
        window_list = [0, 20, 40, 60]
        df_scaled = self.get_standard_data()
        X_train, y_train, X_val, y_val, X_test, y_test = self.get_data_to_model()
        pred_val = model.predict_interval(X_val, self.conf.n_future)
        df_val = df_scaled.iloc[
                 self.conf.train_split - self.conf.window_size: self.conf.val_split, :
                 ]
        date_val = df_val.index.values
        for ind, val in enumerate(window_list):
            assert X_val.shape[0] - 1 >= val, f"Окна c номером {val} не существует"
            y_val_inv, pred_val_inv = self.get_inverse_values(
                y_val[val, :, :], pred_val[val, :, :]
            )
            y_based_pred = self.scaler.inverse_transform(X_val[val, :, :])
            ax[ind].plot(
                date_val[val: val + self.conf.window_size],
                y_based_pred[:, -1],
                color="cornflowerblue",
            )
            ax[ind].plot(
                date_val[
                val + self.conf.window_size:
                val + self.conf.window_size + self.conf.n_future
                ],
                y_val_inv[:, -1],
                label="Факт",
                color="cornflowerblue",
            )
            ax[ind].plot(
                date_val[
                val + self.conf.window_size:
                val + self.conf.window_size + self.conf.n_future
                ],
                pred_val_inv[:, -1],
                label="Прогноз",
                linestyle="--",
                color="salmon",
            )
            ax[ind].axvline(
                date_val[val + self.conf.window_size], color="k", linestyle="--"
            )
            ax[ind].axvspan(
                date_val[val + self.conf.window_size],
                date_val[val + self.conf.window_size + self.conf.n_future],
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
        plt.show()

    def plot_test_window(self, model: Any) -> None:
        """
        Plot 2 test window
        :param model: RNN model
        :return:
        """
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax = ax.ravel()
        window_list = [0, 20]
        df_scaled = self.get_standard_data()
        X_train, y_train, X_val, y_val, X_test, y_test = self.get_data_to_model()
        pred_test = model.predict_interval(X_test, self.conf.n_future)
        df_test = df_scaled.iloc[self.conf.val_split - self.conf.window_size:, :]
        date_test = df_test.index.values
        for ind, val in enumerate(window_list):
            y_test_inv, pred_test_inv = self.get_inverse_values(
                y_test[val, :, :], pred_test[val, :, :]
            )
            y_based_pred = self.scaler.inverse_transform(X_test[val, :, :])
            ax[ind].plot(
                date_test[val: val + self.conf.window_size],
                y_based_pred[:, -1],
                color="cornflowerblue",
            )
            ax[ind].plot(
                date_test[
                val + self.conf.window_size:
                val + self.conf.window_size + self.conf.n_future
                ],
                y_test_inv[:, -1],
                label="Факт",
                color="cornflowerblue",
            )
            ax[ind].plot(
                date_test[
                val + self.conf.window_size:
                val + self.conf.window_size + self.conf.n_future
                ],
                pred_test_inv[:, -1],
                label="Прогноз",
                linestyle="--",
                color="salmon",
            )
            ax[ind].axvline(
                date_test[val + self.conf.window_size], color="k", linestyle="--"
            )
            ax[ind].axvspan(
                date_test[val + self.conf.window_size],
                date_test[val + self.conf.window_size + self.conf.n_future],
                alpha=0.5,
                color="lightgray",
            )
            mape = (
                    mean_absolute_percentage_error(y_test_inv[:, -1], pred_test_inv[:, -1]) * 100
            )
            ax[ind].set_title(f"Предсказание модели. МАРЕ={round(mape, 2)}.")
            ax[ind].xaxis.set_major_formatter(dates.DateFormatter("%d-%m-%y"))
            ax[ind].tick_params(axis="x", labelrotation=45)
            ax[ind].grid()
            ax[ind].legend()
        plt.show()

    def calc_validation_mape(self, model: Any) -> List[float]:
        """
        Calculate MAPE on all validation windows
        :param model: RNN model
        :return:
        """
        X_train, y_train, X_val, y_val, X_test, y_test = self.get_data_to_model()
        list_num_wind = np.arange(X_val.shape[0])
        list_metric = []
        pred_val = model.predict_interval(X_val, self.conf.n_future)
        for wind in list_num_wind:
            y_val_inv, pred_val_inv = self.get_inverse_values(
                y_val[wind, :, :], pred_val[wind, :, :]
            )
            mape_e1d1 = mean_absolute_percentage_error(
                y_val_inv[:, -1], pred_val_inv[:, -1]
            )
            list_metric.append(mape_e1d1 * 100)
        return list_metric

    def calc_average_validation_mape(self, model: Any) -> float:
        """
        Calculate average MAPE on validation data
        :param model: RNN model
        :return:
        """
        list_metric = self.calc_validation_mape(model)
        return sum(list_metric) / len(list_metric)

    def calc_test_mape(self, model: Any) -> List[float]:
        """
        Calculate MAPE on all test window
        :param model: RNN model
        :return:
        """
        X_train, y_train, X_val, y_val, X_test, y_test = self.get_data_to_model()
        list_num_wind = np.arange(X_test.shape[0])
        list_metric = []
        pred_test = model.predict_interval(X_test, self.conf.n_future)
        for wind in list_num_wind:
            y_test_inv, pred_test_inv = self.get_inverse_values(
                y_test[wind, :, :], pred_test[wind, :, :]
            )
            mape_e1d1 = mean_absolute_percentage_error(
                y_test_inv[:, -1], pred_test_inv[:, -1]
            )
            list_metric.append(mape_e1d1 * 100)
        return list_metric

    def calc_average_test_mape(self, model: Any) -> float:
        """
        Calculate average MAPE on test data
        :param model: RNN model
        :return:
        """
        list_metric = self.calc_test_mape(model)
        return sum(list_metric) / len(list_metric)

    def plot_chart_mape_window(self, model: Any) -> None:
        """
        Plot MAPE on validation and test data
        :param model: model
        :return:
        """
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        ax = ax.ravel()
        val_mape = self.calc_validation_mape(model)
        test_mape = self.calc_test_mape(model)
        x_val = np.arange(len(val_mape)) + 1
        x_test = np.arange(len(test_mape)) + 1
        ax[0].plot(x_val, val_mape)
        ax[0].set_title("Валидационный набор")
        ax[0].set_xlabel("Номер окна")
        ax[0].set_ylabel("MAPE, %")
        ax[0].grid()

        ax[1].plot(x_test, test_mape)
        ax[1].set_title("Тестовый наборе")
        ax[1].set_xlabel("Номер окна")
        ax[1].set_ylabel("MAPE, %")
        ax[1].grid()
        plt.show()
