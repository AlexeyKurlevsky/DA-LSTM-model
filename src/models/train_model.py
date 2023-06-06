import click
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

from models import DualAttentionRNN
from src import seed_everything, Config, WindowGenerator


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("window_size", type=click.INT)
@click.argument("max_epochs", type=click.INT)
@click.argument("output_path", type=click.Path())
def train_model(input_path: str, output_path: str, window_size: int, max_epochs: int) -> None:
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
    early_stopping = EarlyStopping(monitor="val_loss", patience=conf.patience, mode="min")
    checkpoint_filepath = f"./{output_path}/weight/checkpoint"
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    da_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=conf.loss_func)

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

    da_model.save(f'./{output_path}/entire_model/my_model.h5')

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Потери на тренировочном наборе')
    plt.plot(epochs, val_loss, 'r', label='Потери на валидационном наборе')
    plt.title('Значение функции потерь при обучении')
    plt.yscale('log', base=10)
    plt.legend()
    plt.xlabel('Эпохи обучения')
    plt.ylabel('MSE')
    plt.grid()
    plt.savefig('./reports/figures/loss_func.png')


if __name__ == '__main__':
    train_model()
