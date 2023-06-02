import tensorflow as tf
import os

from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

from src.models.da_rnn_model import DualAttentionRNN
from src.config import Config
from src.func import get_data, seed_everything
from src.window_generator import WindowGenerator

seed_everything()
os.system("dvc pull")
path = "./data_yandex.csv"
print("==> Load dataset ...")
df = get_data(path)
df_search = df[["Дата", "Заражений за день", "Day sin", "Day cos", "Характерстика ДБ"]]
df_search = df_search.set_index("Дата")
df_search.dropna(inplace=True)
conf = Config(df_search)
conf.window_size = 90
conf.patience = 10
conf.batch_size = 32
conf.n_future = 1
conf.epochs = 300
w_all_features = WindowGenerator(
    df_search, mean_flg=True, scaler=MinMaxScaler(), conf=conf
)
X_train, y_train, X_val, y_val, X_test, y_test = w_all_features.get_data_to_model()
train_data_multi, val_data_multi = w_all_features.get_tensor_data()
DA_model = DualAttentionRNN(decoder_num_hidden=64, encoder_num_hidden=64, conf=conf)
early_stopping = EarlyStopping(monitor="val_loss", patience=conf.patience, mode="min")
checkpoint_filepath = "./tmp/checkpoint"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
)

DA_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=conf.loss_func)

history = DA_model.fit(train_data_multi, epochs=conf.epochs, validation_data=val_data_multi, verbose=1,
                       callbacks=[early_stopping, model_checkpoint_callback],
                       steps_per_epoch=conf.steps_per_epoch, validation_steps=conf.validation_steps)

DA_model.load_weights(checkpoint_filepath)

conf.n_future = 20
w_one_target = WindowGenerator(
    df_search, mean_flg=True, scaler=MinMaxScaler(), conf=conf
)

val_mape = w_one_target.calc_average_validation_mape(DA_model)
test_mape = w_one_target.calc_average_test_mape(DA_model)
print(f"Val MAPE {round(val_mape, 2)}; Test MAPE: {round(test_mape, 2)}")

w_one_target.plot_test_window(DA_model)
