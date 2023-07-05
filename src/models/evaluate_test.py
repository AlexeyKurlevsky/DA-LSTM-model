import json
import logging

import click
import pandas as pd
import numpy as np
import tensorflow as tf
import yaml

from sklearn.preprocessing import MinMaxScaler

from models import DualAttentionRNN
from src import Config, seed_everything, WindowGenerator
from src.visualization.calculate_metrics import calc_test_metric


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("model_feature_path", type=click.Path())
@click.argument("output_metric_average", type=click.Path())
@click.argument("output_test_metric_all", type=click.Path())
def evaluate_test(
    input_path: str,
    model_feature_path: str,
    output_metric_average: str,
    output_test_metric_all: str,
):
    """
    Function for predict values.
    :param input_path: path processed data
    :param model_feature_path: directory with model
    :param output_test_metric_all: path to save metric on all windows (CSV)
    :param output_metric_average: path to save average metric (JSON)
    """
    logging.basicConfig(level=logging.INFO)
    params = yaml.safe_load(open("params.yaml"))["train"]
    seed_everything(params["seed"])

    logging.info("Data read")
    df_search = pd.read_csv(input_path, parse_dates=["Дата"])
    df_search = df_search.set_index("Дата")
    conf = Config(df_search)
    conf.window_size = params["window_size"]
    conf.batch_size = params["batch_size"]
    conf.n_future = params["n_future"]

    logging.info("Generate window")
    w_one_target = WindowGenerator(
        df_search, mean_flg=True, scaler=MinMaxScaler(), conf=conf
    )
    da_model = DualAttentionRNN(
        decoder_num_hidden=params["num_hidden_state"],
        encoder_num_hidden=params["num_hidden_state"],
        conf=conf,
    )
    checkpoint_filepath = f"./{model_feature_path}/checkpoint"

    da_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=conf.loss_func)
    logging.info("Load weights")
    da_model.load_weights(checkpoint_filepath)

    X_train, y_train, X_val, y_val, X_test, y_test = w_one_target.get_data_to_model()
    logging.info("Predict interval")
    y_pred = da_model.predict_interval(X_test, w_one_target.conf.n_future)
    mape_arr, rmse_arr = calc_test_metric(w_one_target, y_pred)

    logging.info("Calculate all windows metric")
    df_metric_all_window = pd.DataFrame(data={"MAPE": mape_arr, "RMSE": rmse_arr})
    df_metric_all_window.to_csv(output_test_metric_all, index=False)

    logging.info("Calculate average window metric")
    df_average_metric = {"MAPE": np.average(mape_arr), "RMSE": np.average(rmse_arr)}
    with open(output_metric_average, "w") as validation_score_file:
        json.dump(df_average_metric, validation_score_file, indent=4)

    # plot_validation_window(w_one_target, y_pred, output_figure_path)


if __name__ == "__main__":
    evaluate_test()
