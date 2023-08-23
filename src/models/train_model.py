import logging
import os
import click
import pandas as pd
import tensorflow as tf
import yaml
import mlflow

from dotenv import load_dotenv
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

from src import (
    seed_everything,
    Config,
    WindowGenerator,
)
from src.models.da_rnn_model import DualAttentionRNN


@click.command()
@click.argument(
    "input_path", type=click.Path(), default="./data/processed/data_search.csv"
)
@click.argument("output_model_path", type=click.Path(), default="./models/saved_model")
def train_model(input_path: str, output_model_path: str) -> None:
    """
    Function for train model. When training, the EarlyStopping method is used.
    Training settings are declared in the config dictionary.
    :param input_path: path processed data
    :param output_model_path: path to save model property
    """
    load_dotenv()
    remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(remote_server_uri)

    logging.basicConfig(level=logging.DEBUG)
    params = yaml.safe_load(open("params.yaml"))["train"]
    seed_everything(params["seed"])

    logging.info("Data read")
    df_search = pd.read_csv(input_path, parse_dates=["Дата"])
    df_search = df_search.set_index("Дата")
    dataset = mlflow.data.from_pandas(df_search)

    conf = Config(df_search)
    conf.window_size = params["window_size"]
    conf.patience = params["patience"]
    conf.batch_size = params["batch_size"]
    conf.steps_per_epoch = params["steps_per_epoch"]
    conf.validation_steps = params["validation_steps"]
    conf.epochs = params["max_epochs"]
    conf.n_future = 1

    logging.info("Generate window")
    w_all_features = WindowGenerator(
        df_search, mean_flg=True, scaler=MinMaxScaler(), conf=conf
    )

    X_train, y_train, X_val, y_val, X_test, y_test = w_all_features.get_data_to_model()
    train_data_multi, val_data_multi = w_all_features.get_tensor_data()

    logging.info("Initialize model")
    da_model = DualAttentionRNN(
        decoder_num_hidden=params["num_hidden_state"],
        encoder_num_hidden=params["num_hidden_state"],
        conf=conf,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=conf.patience, mode="min"
    )
    checkpoint_filepath = f"./{output_model_path}/weights/checkpoint"
    weights_dir = f"./{output_model_path}/weights"
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    logging.info("Compile model")
    da_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=conf.loss_func)

    logging.info("Fit model")
    history = da_model.fit(
        train_data_multi,
        epochs=conf.epochs,
        validation_data=val_data_multi,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint_callback],
        steps_per_epoch=conf.steps_per_epoch,
        validation_steps=conf.validation_steps,
    )
    da_model.load_weights(checkpoint_filepath)
    mlflow.set_experiment("train model")
    with mlflow.start_run():
        logging.info("Logging dataset")
        mlflow.log_input(dataset, context="training")
        mlflow.log_artifact(input_path)
        logging.info("Logging metrics")
        for epoch in range(1, params["max_epochs"] + 1):
            mlflow.log_metric(
                key="mse_train_losses",
                value=history.history["loss"][epoch - 1],
                step=epoch,
            )
            mlflow.log_metric(
                key="mse_val_losses",
                value=history.history["val_loss"][epoch - 1],
                step=epoch,
            )
        # Log params
        mlflow.log_params(params)
        # Log data
        mlflow.log_artifact(weights_dir)
        # Log metric
        logging.info("predict test")
        y_pred = da_model.predict(X_test)
        logging.info("Calculate signature")
        model_signature = mlflow.models.infer_signature(X_test, y_pred)
        mlflow.log_artifacts(f"./{output_model_path}/weights")
        logging.info("logging model")
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
    train_model()
