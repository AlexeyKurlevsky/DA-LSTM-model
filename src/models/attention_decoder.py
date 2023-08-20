import tensorflow as tf

from keras.layers import Layer


class AttentionDecoder(Layer):
    def __init__(self, decoder_num_hidden=64, encoder_num_hidden=64):
        """
        Decoder block with temporal attention mechanism
        :param decoder_num_hidden: Number of cells in decoder block
        :param encoder_num_hidden: Number of cells in encoder block
        """
        super(AttentionDecoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden

    def build(self, input_shape):
        """
        Building layer parameters
        :param input_shape: tuple with shape
        """
        self.w = self.add_weight(
            name="w_d",
            shape=(input_shape[-1] - self.encoder_num_hidden, 1),
            initializer="random_normal",
            trainable=True,
        )
        self.u_d = self.add_weight(
            name="u_d",
            shape=(self.encoder_num_hidden, 1),
            initializer="random_normal",
            trainable=True,
        )
        self.v_d = self.add_weight(
            name="v_d", shape=(1, 1), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        """
        Eq. 12 in the article
        :param inputs: (batch_size*window_size, 2 * decoder_num_hidden + encoder_num_hidden)
        :return (batch_size*window_size, 1)
        """
        state = inputs[:, : 2 * self.decoder_num_hidden]
        h_i = inputs[:, 2 * self.decoder_num_hidden :]
        l_i = tf.keras.activations.tanh(
            tf.matmul(state, self.w) + tf.matmul(h_i, self.u_d)
        )
        return tf.matmul(l_i, self.v_d)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'encoder_num_hidden': self.encoder_num_hidden,
            'decoder_num_hidden': self.decoder_num_hidden,
        })
        return config

