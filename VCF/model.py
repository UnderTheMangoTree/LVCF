import tensorflow as tf

def MD():
    def Conv(kernel, x, kind="same"):
        bloack = tf.keras.layers.Conv2D(kernel, (3, 3), strides=(1, 1), padding=kind,
                                        kernel_initializer=tf.random_uniform_initializer)(x)
        bloack = tf.keras.layers.BatchNormalization()(bloack)
        bloack = tf.nn.relu(bloack)
        return bloack
    input_shape = (240, 320, 3)
    base_feature_model = tf.keras.applications.vgg16.VGG16(weights=None, include_top=False,
                                                           input_shape=input_shape)
    x = base_feature_model.layers[6].output
    bloack_3 = Conv(256, x)
    bloack_3 = Conv(256, bloack_3)
    bloack_3 = Conv(256, bloack_3)
    bloack_3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2])(bloack_3)
    bloack_4 = Conv(128, bloack_3)
    bloack_4 = Conv(128, bloack_4)
    bloack_4 = Conv(128, bloack_4)
    bloack_4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2])(bloack_4)
    bloack_5 = Conv(64, bloack_4)
    bloack_5 = Conv(64, bloack_5)
    bloack_5 = Conv(64, bloack_5)
    bloack_5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2])(bloack_5)
    bloack_6 = tf.keras.layers.Flatten()(bloack_5)
    dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.sigmoid)(bloack_6)
    dense1 = tf.keras.layers.Dropout(0.5)(dense1)
    out = tf.keras.layers.Dense(units=210, activation=tf.nn.sigmoid)(dense1)
    out = tf.keras.layers.Dropout(0.5)(out)
    out1 = tf.keras.layers.Dense(units=70, activation=tf.nn.softmax)(dense1)
    out1 = tf.keras.layers.Dropout(0.5)(out1)
    out = tf.reshape(out, [-1, 7, 10, 3])
    out1 = tf.reshape(out1, [-1, 7, 10, 1])
    outs = tf.concat([out, out1], 3)
    model = tf.keras.models.Model(inputs=base_feature_model.input, outputs=outs)
    model.summary()
    return model