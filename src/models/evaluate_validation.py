import json

import click
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from models import DualAttentionRNN
from src import (
    Config,
    seed_everything,
    WindowGenerator,
    calc_test_metric,
    plot_test_window,
)
from src.visualization.calculate_metrics import calc_validation_metric
from src.visualization.plot_window import plot_validation_window


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("model_feature_path", type=click.Path())
@click.argument("window_size", type=click.INT)
@click.argument("n_future", type=click.INT)
@click.argument("output_metric_all", type=click.Path())
@click.argument("output_metric_average", type=click.Path())
@click.argument("output_figure_path", type=click.Path())
def predict_model(
    input_path: str,
    model_feature_path: str,
    output_metric_all: str,
    output_metric_average: str,
    output_figure_path: str,
    window_size: int,
    n_future: int,
):
    """
    Function for predict values.
    :param input_path: path processed data
    :param n_future: forecast horizon
    :param window_size: length of window
    :param model_feature_path: directory with model
    :param output_metric_all: path to save metric on all windows
    :param output_metric_average: path to save average metric
    :param output_figure_path: path to save figure with predicted values
    """
    seed_everything()
    df_search = pd.read_csv(input_path, parse_dates=["Дата"])
    df_search = df_search.set_index("Дата")
    conf = Config(df_search)
    conf.window_size = window_size
    conf.patience = 10
    conf.batch_size = 32
    conf.n_future = n_future
    w_one_target = WindowGenerator(
        df_search, mean_flg=True, scaler=MinMaxScaler(), conf=conf
    )
    da_model = DualAttentionRNN(decoder_num_hidden=64, encoder_num_hidden=64, conf=conf)
    checkpoint_filepath = f"./{model_feature_path}/weight/checkpoint"

    da_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=conf.loss_func)

    da_model.load_weights(checkpoint_filepath)

    X_train, y_train, X_val, y_val, X_test, y_test = w_one_target.get_data_to_model()
    y_pred = da_model.predict_interval(X_val, w_one_target.conf.n_future)
    mape_arr, rmse_arr = calc_validation_metric(w_one_target, y_pred)

    df_metric_all_window = pd.DataFrame(data={"MAPE": mape_arr, "RMSE": rmse_arr})
    df_metric_all_window.to_csv(output_validation_metric_all, index=False)

    df_average_metric = {"MAPE": np.average(mape_arr), "RMSE": np.average(rmse_arr)}
    with open(output_metric_average, "w") as validation_score_file:
        json.dump(df_average_metric, validation_score_file, indent=4)

    # plot_validation_window(w_one_target, y_pred, output_figure_path)

    y_pred = da_model.predict_interval(X_test, w_one_target.conf.n_future)
    mape_arr, rmse_arr = calc_test_metric(w_one_target, y_pred)

    df_metric_all_window = pd.DataFrame(data={"MAPE": mape_arr, "RMSE": rmse_arr})
    df_metric_all_window.to_csv(output_test_metric_all, index=False)

    df_average_metric = {"MAPE": np.average(mape_arr), "RMSE": np.average(rmse_arr)}
    with open(output_metric_average, "w") as validation_score_file:
        json.dump(df_average_metric, validation_score_file, indent=4)

    plot_test_window(w_one_target, y_pred, output_figure_path)


if __name__ == "__main__":
    predict_model()
