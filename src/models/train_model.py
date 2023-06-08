import click
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

from models import DualAttentionRNN
from src import seed_everything, Config, WindowGenerator, calc_validation_metric, plot_validation_window


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("window_size", type=click.INT)
@click.argument("n_future", type=click.INT)
@click.argument("max_epochs", type=click.INT)
@click.argument("output_model_path", type=click.Path())
@click.argument("output_metric_all", type=click.Path())
@click.argument("output_metric_average", type=click.Path())
@click.argument("output_figure_path", type=click.Path())
def train_model(
    input_path: str,
    output_model_path: str,
    output_metric_all: str,
    output_metric_average: str,
    output_figure_path: str,
    window_size: int,
    max_epochs: int,
    n_future: int,
) -> None:
    """
    Function for train model. When training, the EarlyStopping method is used.
    Training settings are declared in the config dictionary.
    :param input_path: path processed data
    :param output_model_path: path to save model property
    :param output_metric_all: path to save metric on all windows
    :param output_metric_average: path to save average metric
    :param output_figure_path: path to save figure with predicted values
    :param window_size: length of window
    :param n_future: forecast horizon
    :param max_epochs: maximum number of epochs
    """
    seed_everything()
    df_search = pd.read_csv(input_path, parse_dates=["Дата"])
    df_search = df_search.set_index("Дата")
    conf = Config(df_search)
    conf.window_size = window_size
    conf.patience = 10
    conf.batch_size = 32
    conf.n_future = 1
    conf.epochs = max_epochs
    w_all_features = WindowGenerator(
        df_search, mean_flg=True, scaler=MinMaxScaler(), conf=conf
    )
    train_data_multi, val_data_multi = w_all_features.get_tensor_data()
    da_model = DualAttentionRNN(decoder_num_hidden=64, encoder_num_hidden=64, conf=conf)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=conf.patience, mode="min"
    )
    checkpoint_filepath = f"./{output_model_path}/weight/checkpoint"
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    da_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=conf.loss_func)

    # da_model.fit(
    #     train_data_multi,
    #     epochs=conf.epochs,
    #     validation_data=val_data_multi,
    #     verbose=1,
    #     callbacks=[early_stopping, model_checkpoint_callback],
    #     steps_per_epoch=conf.steps_per_epoch,
    #     validation_steps=conf.validation_steps,
    # )

    da_model.load_weights(checkpoint_filepath)

    conf.n_future = n_future
    w_one_target = WindowGenerator(
        df_search, mean_flg=True, scaler=MinMaxScaler(), conf=conf
    )

    X_train, y_train, X_val, y_val, X_test, y_test = w_one_target.get_data_to_model()
    y_pred = da_model.predict_interval(X_val, w_one_target.conf.n_future)
    mape_arr, rmse_arr = calc_validation_metric(w_one_target, y_pred)

    df_metric_all_window = pd.DataFrame(data={"MAPE": mape_arr, "RMSE": rmse_arr})
    df_average_metric = pd.DataFrame(
        data={"MAPE": np.average(mape_arr), "RMSE": np.average(rmse_arr)},
        index=[0]
    )

    df_metric_all_window.to_csv(
        output_metric_all, index=False
    )
    df_average_metric.to_csv(output_metric_average, index=False)

    plot_validation_window(w_one_target, y_pred, output_figure_path)


if __name__ == "__main__":
    train_model()
