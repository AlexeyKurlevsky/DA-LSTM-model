import tensorflow as tf
import pandas as pd


class Config:
    def __init__(self, data: pd.DataFrame):
        self.patience = 10
        self.per_train_split = 0.85
        self.per_val_split = 0.95
        self.window_size_MA = 7
        self.n_future = 20
        self.season_day = 200
        self.batch_size = 32
        self.epochs = 50
        self.step = 1
        self.window_size = int(1.25 * max(self.n_future, self.season_day))
        self.n_samples = data.shape[0]
        self.train_split = int(self.n_samples * self.per_train_split)
        self.val_split = int(self.n_samples * self.per_val_split)
        self.num_features = data.shape[1]
        self.seed = 0
        self.metrics = ['mean_squared_error']
        self.steps_per_epoch = 50
        self.validation_steps = 20
        self.loss_func = tf.keras.losses.MeanSquaredError()
