import json
import logging
import os
import click
import mlflow
import pandas as pd
import numpy as np
import tensorflow as tf
import yaml

from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

from src import (
    Config,
    seed_everything,
    WindowGenerator,
)
from src.models.attention_decoder import AttentionDecoder
from src.models.attention_encoder import AttentionEncoder
from src.models.da_rnn_model import DualAttentionRNN
from src.visualization.calculate_metrics import calc_validation_metric
from src.visualization.plot_window import plot_validation_window


@click.command()
@click.argument("input_path", type=click.Path(), default="./data/processed/data_search.csv")
@click.argument("model_feature_path", type=click.Path(), default="./models/saved_model")
def evaluate_validation(input_path: str, model_feature_path: str):
    """
    Function for predict values.
    :param input_path: path processed data
    :param model_feature_path: directory with model
    """
    load_dotenv()
    remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(remote_server_uri)

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
    checkpoint_filepath = f"./{model_feature_path}/weights/checkpoint"

    da_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=conf.loss_func)
    logging.info("Load weights")
    da_model.load_weights(checkpoint_filepath)

    X_train, y_train, X_val, y_val, X_test, y_test = w_one_target.get_data_to_model()

    mlflow.set_experiment("evaluate validation")
    with mlflow.start_run():
        logging.info("Predict interval")
        y_pred = da_model.predict_interval(X_val, w_one_target.conf.n_future)
        model_signature = mlflow.models.infer_signature(X_val, y_pred)
        mape_arr, rmse_arr = calc_validation_metric(w_one_target, y_pred)

        logging.info("Calculate all windows metric")
        df_metric_all_window = pd.DataFrame(data={"MAPE": mape_arr, "RMSE": rmse_arr})
        df_metric_all_window.to_csv(
            "./reports/validation_metric/metric_all_window.csv", index=False
        )

        logging.info("Calculate average window metric")
        df_average_metric = {"MAPE": np.average(mape_arr), "RMSE": np.average(rmse_arr)}
        with open(
            "./reports/validation_metric/average_metric.json", "w"
        ) as validation_score_file:
            json.dump(df_average_metric, validation_score_file, indent=4)

        figure = plot_validation_window(
            w_one_target, y_pred, "./reports/figures/validation_predict_dvc.png"
        )
        # log dataset
        mlflow.log_input(dataset, context="validation")
        # Log params
        mlflow.log_params(params)
        # Log data
        mlflow.log_artifact(input_path)
        # Log weights
        mlflow.log_artifact(f"./{model_feature_path}/weights", artifact_path="da_model")
        mlflow.log_artifact("params.yaml", artifact_path="da_model")
        # Log metric
        mlflow.log_metric("rmse", np.average(rmse_arr))
        mlflow.log_metric("mape", np.average(mape_arr))
        # Log plot
        mlflow.log_figure(figure, "./reports/figures/validation_predict_dvc.png")
        # log model
        mlflow.tensorflow.log_model(
            model=da_model,
            artifact_path="da_model",
            signature=model_signature,
            registered_model_name="da_model",
            custom_objects={"AttentionEncoder": AttentionEncoder,
                            "AttentionDecoder": AttentionDecoder},
            code_paths=[
                "src/models/attention_decoder.py",
                "src/models/attention_encoder.py",
                "src/models/da_rnn_model.py",
            ],
        )


if __name__ == "__main__":
    evaluate_validation()
