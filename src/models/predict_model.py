import click
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from models import DualAttentionRNN
from src import Config, seed_everything, WindowGenerator, plot_test_window


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("n_future", type=click.INT)
@click.argument("model_feature_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def predict_model(input_path: str, n_future: int, model_feature_path: str, output_path: str):
    seed_everything()
    df_search = pd.read_csv(input_path, parse_dates=["Дата"])
    df_search = df_search.set_index("Дата")
    conf = Config(df_search)
    conf.window_size = 90
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

    plot_test_window(w_one_target, da_model, output_path)


if __name__ == '__main__':
    predict_model()
