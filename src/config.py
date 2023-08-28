import tensorflow as tf
import pandas as pd


class Config:
    def __init__(self, data: pd.DataFrame = None):
        """
        Dictionary with research settings
        :param data: dataframe for research
        """
        # TODO: fix config
        if data is not None:
            self.per_train_split = 0.85  # Percent for train split
            self.per_val_split = 0.95  # Percent for validation split
            self.n_samples = data.shape[0]  # Total number of observations
            self.train_split = int(
                self.n_samples * self.per_train_split
            )  # Index of train split
            self.val_split = int(
                self.n_samples * self.per_val_split
            )  # Index of validation split
            self.num_features = data.shape[1]  # Number of time series
        else:
            self.num_features = 4
        self.patience = 10  # Patience for EarlyStopping.
        self.window_size_MA = 7  # Size of the moving window
        self.n_future = 20  # Forecast horizon
        self.batch_size = 32  # Number of window in batch
        self.epochs = 50  # Number of epochs to train the model
        self.window_size = 90  # Number of observations in the window
        self.seed = 0  # Seed for random
        self.steps_per_epoch = 50  # Number of steps in training in one epoch
        self.validation_steps = 20  # Number of steps in validation in one epoch
        self.loss_func = (
            tf.keras.losses.MeanSquaredError()
        )  # Loss function for training model
