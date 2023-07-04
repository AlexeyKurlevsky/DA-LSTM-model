import logging
import click
import pandas as pd
import tensorflow as tf
import yaml

from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

from models import DualAttentionRNN
from src import (
    seed_everything,
    Config,
    WindowGenerator,
)


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_model_path", type=click.Path())
def train_model(input_path: str, output_model_path: str) -> None:
    """
    Function for train model. When training, the EarlyStopping method is used.
    Training settings are declared in the config dictionary.
    :param input_path: path processed data
    :param output_model_path: path to save model property
    """
    logging.basicConfig(level=logging.INFO)
    params = yaml.safe_load(open("params.yaml"))["train"]
    seed_everything(params["seed"])
    df_search = pd.read_csv(input_path, parse_dates=["Дата"])
    df_search = df_search.set_index("Дата")
    logging.info("Data read")
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
    train_data_multi, val_data_multi = w_all_features.get_tensor_data()

    logging.info("Initialize model")
    da_model = DualAttentionRNN(decoder_num_hidden=params["num_hidden_state"], encoder_num_hidden=params["num_hidden_state"], conf=conf)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=conf.patience, mode="min"
    )
    checkpoint_filepath = f"./{output_model_path}/checkpoint"
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
    da_model.fit(
        train_data_multi,
        epochs=conf.epochs,
        validation_data=val_data_multi,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint_callback],
        steps_per_epoch=conf.steps_per_epoch,
        validation_steps=conf.validation_steps,
    )


if __name__ == "__main__":
    train_model()
