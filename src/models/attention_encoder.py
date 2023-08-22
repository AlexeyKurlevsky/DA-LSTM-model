import tensorflow as tf

from typing import Tuple
from keras.layers import Layer


class AttentionEncoder(Layer):
    def __init__(self, encoder_num_hidden: int = 64):
        """
        Encoder block with input attention mechanism
        :param encoder_num_hidden: Number of cells in encoder block
        """
        super(AttentionEncoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden

    def build(self, input_shape):
        """
        Building layer parameters
        :param input_shape: tuple with shape
        """
        self.w = self.add_weight(
            name="w_e",
            shape=(2 * self.encoder_num_hidden, 1),
            initializer="random_normal",
            trainable=True,
        )
        self.u_e = self.add_weight(
            name="u_e",
            shape=(input_shape[-1] - 2 * self.encoder_num_hidden, 1),
            initializer="random_normal",
            trainable=True,
        )
        self.v_e = self.add_weight(
            name="v_e", shape=(1, 1), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        """
        Eq. 8 in the article
        :param inputs: (batch_size*num_features, 2 * encoder_num_hidden + window_size)
        :return (batch_size*num_features, 1)
        """
        state = inputs[:, : 2 * self.encoder_num_hidden]
        x_k = inputs[:, 2 * self.encoder_num_hidden :]
        e_k = tf.keras.activations.tanh(
            tf.matmul(state, self.w) + tf.matmul(x_k, self.u_e)
        )
        return tf.matmul(e_k, self.v_e)

    def get_config(self):
        config = super().get_config().copy()
        config.update({"encoder_num_hidden": self.encoder_num_hidden})
        return config
