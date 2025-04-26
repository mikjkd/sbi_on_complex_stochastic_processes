import tensorflow as tf
def Architecture(lstm_input_shape, gmm_input_shape, embedding_space, num_model):
    seq_input = tf.keras.Input(shape=lstm_input_shape, name='lstm_input')
    lstm_out = tf.keras.layers.LSTM(units=embedding_space, name='lstm_layer')(seq_input)
    gmm_output = tf.keras.Input(shape=gmm_input_shape, name='gmm_shape')
    gmm_flat = tf.keras.layers.Flatten()(gmm_output)
    concat = tf.keras.layers.Concatenate(name='concat_layer')([lstm_out, gmm_flat])
    ann = tf.keras.layers.Dense(num_model)(concat)
    output = tf.keras.layers.Softmax(name='output')(ann)
    model = tf.keras.Model(inputs=[seq_input, gmm_output], outputs=output, name='LSTM_GMM_ANN')
    return model