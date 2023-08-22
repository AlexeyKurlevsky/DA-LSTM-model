import json
import logging

import click
import mlflow
import pandas as pd
import numpy as np
import tensorflow as tf
import yaml

from sklearn.preprocessing import MinMaxScaler

from src import Config, seed_everything, WindowGenerator
from src.visualization.calculate_metrics import calc_test_metric
from src.visualization.plot_window import plot_test_window
from src.models.da_rnn_model import DualAttentionRNN


@click.command()
@click.argument(
    "input_path", type=click.Path(), default="./data/processed/data_search.csv"
)
@click.argument(
    "model_feature_path", type=click.Path(), default="./models/saved_model/"
)
def evaluate_test(
    input_path: str,
    model_feature_path: str,
):
    """
    Function for predict values.
    :param input_path: path processed data
    :param model_feature_path: directory with model
    """
    logging.basicConfig(level=logging.INFO)
    params = yaml.safe_load(open("params.yaml"))["train"]
    seed_everything(params["seed"])

    logging.info("Data read")
    df_search = pd.read_csv(input_path, parse_dates=["Дата"])
    df_search = df_search.set_index("Дата")
    dataset = mlflow.data.from_pandas(df_search)

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
    checkpoint_filepath = f"./{model_feature_path}/weight/checkpoint"

    da_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=conf.loss_func)
    logging.info("Load weights")
    da_model.load_weights(checkpoint_filepath)

    X_train, y_train, X_val, y_val, X_test, y_test = w_one_target.get_data_to_model()

    mlflow.set_experiment("evaluate test")
    with mlflow.start_run():
        logging.info("Predict interval")
        y_pred = da_model.predict_interval(X_test, w_one_target.conf.n_future)
        mape_arr, rmse_arr = calc_test_metric(w_one_target, y_pred)

        logging.info("Calculate all windows metric")
        df_metric_all_window = pd.DataFrame(data={"MAPE": mape_arr, "RMSE": rmse_arr})
        df_metric_all_window.to_csv(
            "./reports/test_metric/metric_all_window.csv", index=False
        )

        logging.info("Calculate average window metric")
        df_average_metric = {"MAPE": np.average(mape_arr), "RMSE": np.average(rmse_arr)}
        with open(
            "./reports/test_metric/average_metric.json", "w"
        ) as validation_score_file:
            json.dump(df_average_metric, validation_score_file, indent=4)
        # log dataset
        mlflow.log_input(dataset, context="test")
        # Log params
        mlflow.log_params(params)
        # Log data
        mlflow.log_artifact(input_path)
        # Log metric
        mlflow.log_metric("rmse", np.average(rmse_arr))
        mlflow.log_metric("mape", np.average(mape_arr))
        # Log figure
        figure = plot_test_window(
            w_one_target, y_pred, "./reports/figures/test_predict_dvc.png"
        )
        mlflow.log_figure(figure, "./reports/figures/test_predict_dvc.png")
        # log model
        model_signature = mlflow.models.infer_signature(X_test, y_pred)
        mlflow.tensorflow.log_model(
            da_model,
            "da_model",
            signature=model_signature,
            code_paths=[
                "src/models/attention_decoder.py",
                "src/models/attention_encoder.py",
                "src/models/da_rnn_model.py",
            ],
        )


if __name__ == "__main__":
    evaluate_test()
