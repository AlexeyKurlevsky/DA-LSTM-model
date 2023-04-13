import tensorflow as tf

from keras.layers import Dense
from src.models.attention_decoder import AttentionDecoder
from src.models.attention_encoder import AttentionEncoder

tf.config.run_functions_eagerly(True)


def _initialize_hidden_state(inputs, num_hidden):
    return [tf.Variable(tf.zeros((inputs.shape[0], num_hidden))),
            tf.Variable(tf.zeros((inputs.shape[0], num_hidden)))]


class DualAttentionRNNModelOnePoint(tf.keras.Model):
    def __init__(self, conf, encoder_num_hidden, decoder_num_hidden):
        super().__init__()
        self.conf = conf
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.lstm_layer_encoder = tf.keras.layers.LSTMCell(self.encoder_num_hidden)
        self.lstm_layer_decoder = tf.keras.layers.LSTMCell(self.decoder_num_hidden)
        self.encoder_attn = AttentionEncoder(encoder_num_hidden=self.encoder_num_hidden)
        self.decoder_attn = AttentionDecoder(decoder_num_hidden=self.decoder_num_hidden,
                                             encoder_num_hidden=self.encoder_num_hidden)
        self.fc = Dense(units=self.decoder_num_hidden, activation=None)
        self.fc_final = Dense(units=self.conf.num_features, activation=None)

    def encoder(self, inputs):
        h_n, s_n = _initialize_hidden_state(inputs, self.encoder_num_hidden)
        batch_size = inputs.shape[0]
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

            x = tf.concat([h_n_exp, s_n_exp, x_t],
                          axis=2)  # => (batch_size, num_features, 2 * encoder_num_hidden + window_size)

            x = self.encoder_attn(
                tf.reshape(x, shape=(-1, 2 * self.encoder_num_hidden + window_size)))  # => (batch_size*num_feature, 1)
            alpha = tf.nn.softmax(tf.reshape(x, shape=(-1, num_features)), axis=1)  # => (batch_size, num_features)
            x_tilde = alpha * inputs[:, t, :]  # => (batch_size, num_features)

            _, finale_state = self.lstm_layer_encoder(x_tilde, states=[h_n, s_n])
            h_n = finale_state[0]
            s_n = finale_state[1]
            h_encoded.append(h_n)

        X_encoded = tf.stack(h_encoded)
        # X_encoded: (batch_size, window_size, encoder_num_hidden)
        return tf.transpose(X_encoded, perm=[1, 0, 2])

    def decoder(self, x_encoded, inputs):
        d_n, c_n = _initialize_hidden_state(inputs, self.decoder_num_hidden)
        window_size = inputs.shape[1]
        y_prev = inputs[:, :, -1]  # => (batch_size, window_size)

        for t in range(window_size):
            d_n_exp = tf.expand_dims(d_n, axis=0)
            d_n_exp = tf.repeat(d_n_exp, repeats=window_size, axis=0)
            d_n_exp = tf.transpose(d_n_exp, perm=[1, 0, 2])

            c_n_exp = tf.expand_dims(c_n, axis=0)
            c_n_exp = tf.repeat(c_n_exp, repeats=window_size, axis=0)
            c_n_exp = tf.transpose(c_n_exp, perm=[1, 0, 2])

            x = tf.concat(values=[d_n_exp, c_n_exp, x_encoded],
                          axis=2)  # => (batch_size, window_size, 2 * decoder_num_hidden + encoder_num_hidden)

            x = self.decoder_attn(tf.reshape(x, shape=(
                -1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)))  # => (batch_size*window_size, 1)

            beta = tf.nn.softmax(tf.reshape(x, shape=(-1, window_size)), axis=1)  # (batch_size, window_size)
            beta = tf.expand_dims(beta, axis=1)  # => (batch_size, 1, window_size)
            context = tf.matmul(beta, x_encoded)[:, 0, :]  # => (batch_size, encoder_num_hidden)

            y_tilde = self.fc(tf.concat((context, tf.expand_dims(y_prev[:, t], axis=1)),
                                        axis=1))  # => (batch_size, encoder_num_hidden)
            _, final_states = self.lstm_layer_decoder(y_tilde, states=[d_n, c_n])

            d_n = final_states[0]
            c_n = final_states[1]

        y_pred = self.fc_final(tf.concat((d_n, context), axis=1))  # => (batch_size, n_future)
        return y_pred

    def call(self, inputs):
        X_encoded = self.encoder(inputs)
        y_pred = self.decoder(X_encoded, inputs)
        y_pred = tf.expand_dims(y_pred, axis=1)
        return y_pred

    def predict_interval(self, inputs, interval):
        batch_size = inputs.shape[0]
        pred = self.predict(inputs)
        predictions = pred
        inputs = tf.concat([inputs[:, 1:self.conf.window_size, :], pred], axis=1)
        for i in range(1, interval):
            pred = self.predict(inputs)
            predictions = tf.concat([predictions, pred], axis=1)
            inputs = tf.concat([inputs[:, 1:self.conf.window_size, :], pred], axis=1)

        return predictions[:, :, -1]
