import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Conv2D, Activation, Multiply, Concatenate
import keras.ops as K


def channel_attention(x, ratio=8):
    ch = x.shape[-1]

    avg_pool = GlobalAveragePooling2D()(x)
    avg_pool = Reshape((1, 1, ch))(avg_pool)

    max_pool = GlobalMaxPooling2D()(x)
    max_pool = Reshape((1, 1, ch))(max_pool)

    mlp = tf.keras.Sequential([
        Dense(ch // ratio, activation='relu'),
        Dense(ch)
    ])

    avg_out = mlp(avg_pool)
    max_out = mlp(max_pool)

    scale = Activation("sigmoid")(avg_out + max_out)
    return Multiply()([x, scale])


def spatial_attention(x):
    avg_pool = K.mean(x, axis=-1, keepdims=True)
    max_pool = K.max(x, axis=-1, keepdims=True)

    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    spatial = Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)
    return Multiply()([x, spatial])


def cbam(x):
    x = channel_attention(x)
    x = spatial_attention(x)
    return x

