import tensorflow as tf
import numpy as np

from typing import Any
from keras.layers import Dense, RNN

from src.models.attention_decoder import AttentionDecoder
from src.models.attention_encoder import AttentionEncoder

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


class DualAttentionRNN(tf.keras.Model):
    def __init__(
        self, conf: Any, encoder_num_hidden: int = 64, decoder_num_hidden: int = 64
    ):
        """
        Class with Dual-Stage Attention-Based Recurrent Neural Network.
        The model belongs to the encoder-decoder type. The encoder block uses input attention with LSTM layer.
        The decoder block uses temporal attention mechanism with LSTM layer.
        :param conf: dictionary with research settings
        :param encoder_num_hidden: Number of cells in encoder block
        :param decoder_num_hidden: Number of cells in decoder block
        """
        super().__init__()
        self.conf = conf
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.lstm_cell_encoder = tf.keras.layers.LSTMCell(self.encoder_num_hidden)
        self.lstm_cell_decoder = tf.keras.layers.LSTMCell(self.decoder_num_hidden)
        self.lstm_layer_encoder = RNN(self.lstm_cell_encoder, return_state=True)
        self.lstm_layer_decoder = RNN(self.lstm_cell_decoder, return_state=True)
        self.encoder_attn = AttentionEncoder(encoder_num_hidden=self.encoder_num_hidden)
        self.decoder_attn = AttentionDecoder(
            decoder_num_hidden=self.decoder_num_hidden,
            encoder_num_hidden=self.encoder_num_hidden,
        )
        self.fc = Dense(units=self.decoder_num_hidden, activation=None)
        self.fc_final = Dense(units=self.conf.num_features, activation=None)

    def encoder(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Encoder block
        :param inputs: batch with input data (batch_size, window_size, num_feature)
        :return x_encoded: encoded information (batch_size, window_size, encoder_num_hidden)
        """
        h_n, s_n = self.lstm_layer_encoder.get_initial_state(inputs)
        window_size = inputs.shape[1]
        num_features = inputs.shape[2]

        h_encoded = []

        for t in range(window_size):
            h_n_exp = tf.expand_dims(h_n, axis=0)
            h_n_exp = tf.repeat(h_n_exp, repeats=num_features, axis=0)
            h_n_exp = tf.transpose(h_n_exp, perm=[1, 0, 2])

            s_n_exp = tf.expand_dims(s_n, axis=0)
            s_n_exp = tf.repeat(s_n_exp, repeats=num_features, axis=0)
            s_n_exp = tf.transpose(s_n_exp, perm=[1, 0, 2])

            x_t = tf.transpose(inputs, perm=[0, 2, 1])

            x = tf.concat(
                [h_n_exp, s_n_exp, x_t], axis=2
            )  # => (batch_size, num_features, 2 * encoder_num_hidden + window_size)

            x = self.encoder_attn(
                tf.reshape(x, shape=(-1, 2 * self.encoder_num_hidden + window_size))
            )  # => (batch_size*num_feature, 1)
            alpha = tf.nn.softmax(
                tf.reshape(x, shape=(-1, num_features)), axis=1
            )  # => (batch_size, num_features)
            x_tilde = alpha * inputs[:, t, :]  # => (batch_size, num_features)

            _, finale_state = self.lstm_cell_encoder(x_tilde, states=[h_n, s_n])
            h_n = finale_state[0]
            s_n = finale_state[1]
            h_encoded.append(h_n)

        x_encoded = tf.stack(h_encoded)
        return tf.transpose(
            x_encoded, perm=[1, 0, 2]
        )  # (batch_size, window_size, encoder_num_hidden)

    def decoder(self, x_encoded: tf.Tensor, inputs: tf.Tensor) -> tf.Tensor:
        """
        Decoder block
        :param x_encoded: Information from encoder (batch_size, window_size, encoder_num_hidden))
        :param inputs: batch with input data (batch_size, window_size, num_feature)
        :return y_pred: predicted values
        """
        d_n, c_n = self.lstm_layer_decoder.get_initial_state(inputs)
        window_size = inputs.shape[1]
        y_prev = inputs[:, :, -1]  # => (batch_size, window_size)

        for t in range(window_size):
            d_n_exp = tf.expand_dims(d_n, axis=0)
            d_n_exp = tf.repeat(d_n_exp, repeats=window_size, axis=0)
            d_n_exp = tf.transpose(d_n_exp, perm=[1, 0, 2])

            c_n_exp = tf.expand_dims(c_n, axis=0)
            c_n_exp = tf.repeat(c_n_exp, repeats=window_size, axis=0)
            c_n_exp = tf.transpose(c_n_exp, perm=[1, 0, 2])

            x = tf.concat(
                values=[d_n_exp, c_n_exp, x_encoded], axis=2
            )  # => (batch_size, window_size, 2 * decoder_num_hidden + encoder_num_hidden)

            x = self.decoder_attn(
                tf.reshape(
                    x, shape=(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)
                )
            )  # => (batch_size*window_size, 1)

            beta = tf.nn.softmax(
                tf.reshape(x, shape=(-1, window_size)), axis=1
            )  # (batch_size, window_size)
            beta = tf.expand_dims(beta, axis=1)  # => (batch_size, 1, window_size)
            context = tf.matmul(beta, x_encoded)[
                :, 0, :
            ]  # => (batch_size, encoder_num_hidden)

            y_tilde = self.fc(
                tf.concat((context, tf.expand_dims(y_prev[:, t], axis=1)), axis=1)
            )  # => (batch_size, encoder_num_hidden)
            _, final_states = self.lstm_cell_decoder(y_tilde, states=[d_n, c_n])

            d_n = final_states[0]
            c_n = final_states[1]

        y_pred = self.fc_final(
            tf.concat((d_n, context), axis=1)
        )  # => (batch_size, n_future)
        return y_pred

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Method for training the model
        :param inputs: batch with input data (batch_size, window_size, num_feature)
        :return y_pred: predicted values (batch_size, 1, num_feature)
        """
        x_encoded = self.encoder(inputs)
        y_pred = self.decoder(x_encoded, inputs)
        y_pred = tf.expand_dims(y_pred, axis=1)
        return y_pred

    def predict_interval(self, inputs: np.ndarray, interval: int = 20) -> np.ndarray:
        """
        Method for interval prediction.
        The predicted values at time t are used to predict the value t+1
        :param inputs: batch with input data (batch_size, window_size, num_feature)
        :param interval: Forecast horizon
        :return predictions: (batch_size, n_future, num_feature)
        """
        pred = self.predict(inputs)
        predictions = pred
        inputs = np.concatenate([inputs[:, 1 : self.conf.window_size, :], pred], axis=1)
        for i in range(1, interval):
            pred = self.predict(inputs)
            predictions = np.concatenate([predictions, pred], axis=1)
            inputs = np.concatenate(
                [inputs[:, 1 : self.conf.window_size, :], pred], axis=1
            )
        return predictions

    def get_config(self):
        return {
            "encoder_num_hidden": self.encoder_num_hidden,
            "decoder_num_hidden": self.encoder_num_hidden,
            "conf": self.conf.__dict__,
        }
