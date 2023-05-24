import tensorflow as tf

from keras.layers import Layer


class AttentionDecoder(Layer):
    def __init__(self, decoder_num_hidden=128, encoder_num_hidden=128):
        super(AttentionDecoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden

    def build(self, input_shape):
        self.w = self.add_weight(
            name="w_d",
            shape=(input_shape[-1]-self.encoder_num_hidden, 1),
            initializer="random_normal",
            trainable=True,
        )
        self.u_d = self.add_weight(
            name="u_d",
            shape=(self.encoder_num_hidden, 1),
            initializer="random_normal",
            trainable=True
        )
        self.v_d = self.add_weight(
            name="v_d",
            shape=(1, 1),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        """
        Args:
            inputs: (batch_size*window_size, 2 * decoder_num_hidden + encoder_num_hidden)
        """
        state = inputs[:, :2*self.decoder_num_hidden]
        h_i = inputs[:, 2*self.decoder_num_hidden:]
        l_i = tf.keras.activations.tanh(tf.matmul(state, self.w) + tf.matmul(h_i, self.u_d))
        return tf.matmul(l_i, self.v_d)