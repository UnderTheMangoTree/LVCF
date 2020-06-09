import tensorflow as tf
def model2_3():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(100,activation='linear',input_shape = (2,2),
                                   return_sequences = False))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units = 2,activation='linear'))
    model.add(tf.keras.layers.Flatten())
    model.summary()
    return model

