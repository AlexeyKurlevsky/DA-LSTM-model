import tensorflow as tf

from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from src.models.da_rnn_model_interval import DualAttentionRNNModelInterval
from src.window_generator import WindowGenerator
from src.config import Config
from src.func import get_data, seed_everything

path = './data_yandex.csv'
print("==> Load dataset ...")
df = get_data(path)
df_search = df[['Дата', 'Заражений за день', 'Day sin', 'Day cos', 'Характерстика ДБ']]
df_search = df_search.set_index('Дата')
df_search.dropna(inplace=True)
conf = Config(df_search)
seed_everything(conf.seed)
conf.window_size = 60
conf.patience = 10
conf.batch_size = 32
conf.epochs = 2
print("==> Create windows ...")
window_generator = WindowGenerator(df_search, mean_flg=True, scaler=MinMaxScaler(), conf=conf)
X_train, y_train, X_val, y_val, X_test, y_test = window_generator.get_data_to_model()
train_data_multi, val_data_multi = window_generator.get_tensor_data()
DA_model = DualAttentionRNNModelInterval(decoder_num_hidden=32, encoder_num_hidden=32, conf=conf)
early_stopping = EarlyStopping(monitor='val_loss',
                               patience=conf.patience,
                               mode='min')
checkpoint_filepath = './tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

DA_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=conf.loss_func, metrics=conf.metrics)
print("==> Fit model ...")
history = DA_model.fit(train_data_multi, epochs=conf.epochs, validation_data=val_data_multi, verbose=1,
                       callbacks=[early_stopping, model_checkpoint_callback],
                       steps_per_epoch=conf.steps_per_epoch, validation_steps=conf.validation_steps)
DA_model.load_weights(checkpoint_filepath)

val_mape = window_generator.calc_average_validation_mape(DA_model)
test_mape = window_generator.calc_average_test_mape(DA_model)

print(f'Validation MAPE: {round(val_mape, 2)}')
print(f'Test MAPE: {round(test_mape, 2)}')


window_generator.plot_chart_mape_window(DA_model)