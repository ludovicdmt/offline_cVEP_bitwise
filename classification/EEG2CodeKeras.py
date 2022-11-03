from __future__ import division
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    InputLayer,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Permute,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    LeakyReLU,
    Activation,
    SeparableConv2D,
    DepthwiseConv2D,
    SpatialDropout2D,
    Softmax,
    Add,
    GlobalAveragePooling2D,
    concatenate
)
from tensorflow_addons.layers import GELU, Sparsemax

from _utils import positional_encoding

np.random.seed(seed=42)

def basearchi(windows_size, n_channel_input):

    dropoutRate = 0.5
    leak = 0.3
    dropoutType = "Dropout"
    poolType = "max"
    factor_time = 1
    act = "Leaky"
    out = "softmax"

    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout

    if poolType == "avg":
        poolType = AveragePooling2D
    elif poolType == "max":
        poolType = MaxPooling2D

    if act == "GELU":
        act = GELU()
    else:
        act = LeakyReLU(alpha=leak)

    if out == "sparsemax":
        out = Sparsemax()
    else:
        out = Softmax()

    # construct sequential model
    model = Sequential()
    model.add(InputLayer(input_shape=(1, n_channel_input, windows_size)))
    # layer 1
    model.add(
        Conv2D(
            int(16 * factor_time),
            kernel_size=(n_channel_input, 1),
            padding="valid",
            strides=(1, 1),
            data_format="channels_first",
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    #model.add(act())
    model.add(poolType(pool_size=(2, 1), strides=(2, 1), padding="same"))
    model.add(dropoutType(dropoutRate))
    # layer2
    model.add(
        Conv2D(
            int(8 * factor_time),
            kernel_size=(1, 32),
            dilation_rate=(1, 2),
            data_format="channels_first",
            padding="same",
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act)
    model.add(poolType(pool_size=(1, 2), strides=(1, 2), padding="same"))
    model.add(dropoutType(dropoutRate))
    # layer3
    model.add(
        Conv2D(
            int(4 * factor_time),
            kernel_size=(5, 5),
            dilation_rate=(2, 2),
            data_format="channels_first",
            padding="same",
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act)
    model.add(poolType(pool_size=(2, 2), data_format="channels_first", padding="same"))
    model.add(dropoutType(dropoutRate))
    # layer4
    model.add(Flatten())
    model.add(Dense(int(256*factor_time), activation=None))
    model.add(act)
    # layer5
    # model.add(Dense(int(64*factor_time), activation=None))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Dropout(0.5))
    # layer6
    model.add(Dense(2, name="preds", activation="softmax"))
    return model


def basearchiv2(
    windows_size,
    n_channel_input,
    F1=32,
    F2=64,
    D=2,
    dropoutType="Dropout",
    dropoutRate=0.5,
    act="GELU",
    out="sparsemax",
):

    leak = 0.3

    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout

    if act == "GELU":
        act = GELU()
    else:
        act = LeakyReLU(alpha=leak)

    if out == "sparsemax":
        out = Sparsemax()
    else:
        out = Softmax()

    # construct sequential model
    model = Sequential()
    model.add(InputLayer(input_shape=(1, n_channel_input, windows_size)))
    #model.add(Permute((2, 3, 1)))  # Change from channel first to NHWC

    # layer1
    # model.add(
    #     Conv2D(
    #         filters=F1,
    #         kernel_size=(64, 1),
    #         padding="same",
    #         input_shape=(1, n_channel_input, windows_size),
    #         #depth_multiplier=D,
    #         data_format = 'channels_first',
    #         dilation_rate=(2, 1),
    #         #depthwise_initializer="he_uniform",
    #         kernel_initializer="he_uniform",
    #         use_bias=True
    #     )
    # )
    # model.add(BatchNormalization(axis=1, scale=True, center=False))
    # model.add(act())
    # model.add(dropoutType(dropoutRate))

    # layer2
    model.add(
        Conv2D(
            filters=F1,
            kernel_size=(n_channel_input, 1),
            use_bias=True,
            activation=None,
            data_format = 'channels_first',
            kernel_initializer="he_uniform",
        )
    )
    # depthwise_constraint = max_norm(1.)))
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    #model.add(act())
    #model.add(MaxPooling2D((4, 1), data_format="channels_first"))
    model.add(dropoutType(dropoutRate))

    # layer3
    model.add(
        Conv2D(
            F2,
            (1, 30),
            #depthwise_initializer="he_uniform",
            #pointwise_initializer="he_uniform",
            use_bias=True,
            dilation_rate=(1, 3),
            data_format = 'channels_first',
            padding="same",
            kernel_initializer="he_uniform"
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())
    model.add(MaxPooling2D((1, 2), data_format="channels_first"))
    model.add(dropoutType(dropoutRate))

    # layer4
    model.add(
        SeparableConv2D(
            F2//2,
            (5, 5),
            #depthwise_initializer="he_uniform",
            #pointwise_initializer="he_uniform",
            use_bias=True,
            data_format = 'channels_first',
            padding="same",
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())
    model.add(MaxPooling2D((1, 2), data_format="channels_first"))
    model.add(dropoutType(dropoutRate))

    model.add(
        SeparableConv2D(
            F2//4,
            (5, 5),
            #depthwise_initializer="he_uniform",
            #pointwise_initializer="he_uniform",
            use_bias=True,
            data_format = 'channels_first',
            padding="same",
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())
    model.add(MaxPooling2D((1, 2), data_format="channels_first"))
    model.add(dropoutType(dropoutRate))

    # layer 4
    model.add(Flatten())
    model.add(Dense(256, name="dense", activation=None))
    model.add(LeakyReLU(alpha=leak))
    model.add(
        Dense(2, name="output", activation=None)
    )  #  ,kernel_constraint = max_norm(0.25),
    model.add(out)
    return model


def convmixer_block(input, filters, kernel_size):
    """
  Input params
  ------
  input: input tensor
  filters: the number of output channels or filters in pointwise convolution
  kernel_size: kernel_size in depthwise convolution
  """
    shortcut = input

    # Depthwise convolution
    x = DepthwiseConv2D(kernel_size=kernel_size, data_format='channels_first', padding="same")(input)
    x = BatchNormalization(axis=1, scale=True, center=False)(x)
    x = GELU()(x)

    # Shortcut connection
    x = Add()([shortcut, x])

    # Pointwise or 1x1 convolution
    x = Conv2D(filters=filters, kernel_size=1, data_format='channels_first', padding="same")(x)
    x = BatchNormalization(axis=1, scale=True, center=False)(x)
    x = GELU()(x)


    return x


def convMixer(windows_size, n_channel_input, filters=10, dropoutType="Dropout"):

    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout

    dropoutRate = 0.5

    # Defining some hyperparameters for ConvMixer-1536/20
    input_shape = (1, n_channel_input, windows_size)
    patch_size = 3
    filters = 32
    num_classes = 2

    # Input and patch embedding layer
    input = Input(input_shape)
    #x = Permute((3, 2, 1))(input)  # Permute from channel first to NHWC
    x = Conv2D(filters=15, kernel_size=(1, patch_size), data_format='channels_first', strides=(1, patch_size))(
        input
    )  #  Spatial embedding
    x = GELU()(x)
    x = BatchNormalization(axis=1, scale=True, center=False)(x)

    # ConvMixer blocks
    # x = convmixer_block(x, filters, (4, 1))  # Spatial filtering
    # x = dropoutType(dropoutRate)(x)
    x = convmixer_block(x, filters, (1,41))  # Temporal filtering
    x = MaxPooling2D(pool_size=(2, 2), data_format='channels_first', padding="same")(x)
    x = dropoutType(dropoutRate)(x)
    x = convmixer_block(x, filters, 7)
    x = dropoutType(dropoutRate)(x)
    x = convmixer_block(x, filters, 7)
    x = dropoutType(dropoutRate)(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format='channels_first', padding="same")(x)
    x = convmixer_block(x, filters // 2, 5)
    x = dropoutType(dropoutRate)(x)
    x = convmixer_block(x, filters // 2, 3)
    x = dropoutType(dropoutRate)(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format='channels_first', padding="same")(x)

    # Classification head
    # x = GlobalAveragePooling2D(data_format='channels_first',)(x)
    x = Flatten()(x)
    x = Dense(128, activation=None)(x)
    x = LeakyReLU(alpha=0.3)(x)
    # x= Dropout(0.5)(x)
    # # layer5
    # x = Dense(int(64), activation=None)(x)
    # x = LeakyReLU(alpha=0.3)(x)
    outputs = Dense(units=num_classes, activation="softmax")(x)

    convmixer = keras.Model(inputs=input, outputs=outputs)

    return convmixer


def basearchi_patchembedding(windows_size, n_channel_input):
    dropoutRate = 0.5
    leak = 0.3
    dropoutType = "Dropout"
    poolType = "max"
    factor_time = 2
    act = "Leaky"
    out = "softmax"
    patch_size = 3

    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout

    if poolType == "avg":
        poolType = AveragePooling2D
    elif poolType == "max":
        poolType = MaxPooling2D

    if act == "GELU":
        act = GELU
    else:
        act = LeakyReLU

    if out == "sparsemax":
        out = Sparsemax()
    else:
        out = Softmax()

    # construct sequential model
    model = Sequential()

    model.add(InputLayer(input_shape=(1, n_channel_input, windows_size)))

    # Temporal embedding
    model.add(
        Conv2D(
            filters=15,
            data_format="channels_first",
            kernel_size=(1, patch_size),
            strides=(1, patch_size),
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())


    # Spatial fitering
    model.add(
        Conv2D(
            int(16 * factor_time),
            kernel_size=(n_channel_input, 1),
            data_format="channels_first",
            padding="valid",
            strides=(1, 1),
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    #model.add(act())
    # model.add(poolType(pool_size=(2, 1),data_format = 'channels_first',  strides=(2, 1),padding='same'))
    model.add(dropoutType(dropoutRate))

    # Temporal filtering
    model.add(
        Conv2D(
            int(16 * factor_time),
            kernel_size=(1, 20),
            data_format="channels_first",
            padding="same",
            #groups = 8,
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())
    model.add(poolType(pool_size=(2, 2), strides=(2, 2), data_format = 'channels_first', padding='same'))
    model.add(dropoutType(dropoutRate))
    # 2D convo
    model.add(
        Conv2D(
            int(16 * factor_time),
            kernel_size=(5, 5),
            data_format="channels_first",
            padding="same",
            #groups = 8,
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())
    model.add(Permute((1, 3, 2)))
    model.add(poolType(pool_size=(2,2), padding='same')) # data_format = 'channels_first',
    # model.add(dropoutType(dropoutRate))
    # layer4
    model.add(Flatten())
    model.add(Dense(int(128), activation=None))
    model.add(LeakyReLU(alpha=leak))
    # model.add(Dropout(0.5))
    # # layer5
    # model.add(Dense(int(64*factor_time), activation=None))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Dropout(0.5))
    # layer6
    model.add(Dense(2, name="preds", activation=None))
    model.add(out)
    # print(model.summary())
    return model

def basearchi_patchembeddingdilation(windows_size, n_channel_input):
    dropoutRate = 0.5
    leak = 0.3
    dropoutType = "Dropout"
    poolType = "max"
    factor_time = 2
    act = "Leaky"
    out = "softmax"
    patch_size = 3

    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout

    if poolType == "avg":
        poolType = AveragePooling2D
    elif poolType == "max":
        poolType = MaxPooling2D

    if act == "GELU":
        act = GELU
    else:
        act = LeakyReLU

    if out == "sparsemax":
        out = Sparsemax()
    else:
        out = Softmax()

    # construct sequential model
    model = Sequential()

    model.add(InputLayer(input_shape=(1, n_channel_input, windows_size)))

    # Temporal embedding
    model.add(
        Conv2D(
            filters=10,
            data_format="channels_first",
            kernel_size=(1, patch_size),
            strides=(1, patch_size),
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())


    # Spatial fitering
    model.add(
        Conv2D(
            int(16 * factor_time),
            kernel_size=(n_channel_input, 1),
            data_format="channels_first",
            padding="valid",
            strides=(1, 1),
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    #model.add(act())
    # model.add(poolType(pool_size=(2, 1),data_format = 'channels_first',  strides=(2, 1),padding='same'))
    model.add(dropoutType(dropoutRate))

    # Temporal filtering
    model.add(
        Conv2D(
            int(16 * factor_time),
            kernel_size=(1, 20),
            data_format="channels_first",
            padding="same",
            dilation_rate = (1, 2),
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())
    model.add(poolType(pool_size=(2, 2), strides=(2, 2), data_format = 'channels_first', padding='same'))
    model.add(dropoutType(dropoutRate))
    # 2D convo
    model.add(
        Conv2D(
            int(16 * factor_time),
            kernel_size=(5, 5),
            data_format="channels_first",
            padding="same",
            dilation_rate = (2, 2),
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())
    model.add(Permute((1, 3, 2)))
    model.add(poolType(pool_size=(2,2), padding='same'))#, data_format="channels_first"))
    # model.add(dropoutType(dropoutRate))
    # layer4
    model.add(Flatten())
    model.add(Dense(int(128), activation=None))
    model.add(LeakyReLU(alpha=leak))
    # model.add(Dropout(0.5))
    # # layer5
    # model.add(Dense(int(64*factor_time), activation=None))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Dropout(0.5))
    # layer6
    model.add(Dense(2, name="preds", activation=None))
    model.add(out)
    # print(model.summary())
    return model


def basearchi_patchembeddingdepthwise(windows_size, n_channel_input):

    dropoutRate = 0.5
    leak = 0.3
    dropoutType = "Dropout"
    poolType = "max"
    factor_time = 2
    act = "LeakyReLU"
    out = "softmax"
    patch_size = 3

    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout

    if poolType == "avg":
        poolType = AveragePooling2D
    elif poolType == "max":
        poolType = MaxPooling2D

    if act == "GELU":
        act = GELU
    else:
        act = LeakyReLU

    if out == "sparsemax":
        out = Sparsemax()
    else:
        out = Softmax()

    # construct sequential model
    model = Sequential()

    model.add(InputLayer(input_shape=(1, n_channel_input, windows_size)))

    # Temporal embedding
    model.add(
        Conv2D(
            filters=10,
            data_format="channels_first",
            kernel_size=(1, patch_size),
            strides=(1, patch_size),
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())


    # Spatial fitering
    model.add(
        Conv2D(
            int(16 * factor_time),
            kernel_size=(n_channel_input, 1),
            data_format="channels_first",
            padding="valid",
            strides=(1, 1),
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    #model.add(act())
    # model.add(poolType(pool_size=(2, 1),data_format = 'channels_first',  strides=(2, 1),padding='same'))
    model.add(dropoutType(dropoutRate))

    # Temporal filtering
    model.add(
        Conv2D(
            int(16 * factor_time),
            kernel_size=(1, 20),
            data_format="channels_first",
            padding="same",
            #groups = 8,
            kernel_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())
    model.add(poolType(pool_size=(2, 2), strides=(2, 2), data_format = 'channels_first', padding='same'))
    model.add(dropoutType(dropoutRate))
    model.add(Permute((1, 3, 2)))

    # Temporal filtering
    model.add(
        DepthwiseConv2D(
            #int(16 * factor_time),
            kernel_size=(7, 7),
            data_format="channels_first",
            dilation_rate = (2,2),
            padding="same",
            depthwise_initializer="he_uniform",
            #pointwise_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())

    # Temporal filtering
    model.add(
        SeparableConv2D(
            int(16 * factor_time),
            kernel_size=(7, 7),
            data_format="channels_first",
            dilation_rate = (2,2),
            padding="same",
            depthwise_initializer="he_uniform",
            pointwise_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())
    model.add(poolType(pool_size=(2, 2), data_format = 'channels_first', strides=(2, 2), padding='same'))
    model.add(dropoutType(dropoutRate))
    # 2D convo

    model.add(
        SeparableConv2D(
            int(16 * factor_time),
            kernel_size=(5, 5),
            #data_format="channels_first",
            padding="same",
            depthwise_initializer="he_uniform",
            pointwise_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())
    model.add(
        SeparableConv2D(
            int(16 * factor_time),
            kernel_size=(5, 5),
            #data_format="channels_first",
            padding="same",
            depthwise_initializer="he_uniform",
            pointwise_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())
    model.add(poolType(pool_size=(2, 2), padding='same'))

    model.add(
        SeparableConv2D(
            int(16 * factor_time),
            kernel_size=(5, 5),
            #data_format="channels_first",
            padding="same",
            depthwise_initializer="he_uniform",
            pointwise_initializer="he_uniform",
            activation=None,
        )
    )
    model.add(BatchNormalization(axis=1, scale=True, center=False))
    model.add(act())
    model.add(poolType(pool_size=(2, 2), padding='same'))
    # model.add(dropoutType(dropoutRate))
    # layer4
    model.add(Flatten())
    model.add(Dense(int(128), activation=None))
    model.add(LeakyReLU(alpha=leak))
    # model.add(Dropout(0.5))
    # # layer5
    # model.add(Dense(int(128), activation=None))
    # model.add(LeakyReLU(alpha=0.3))
    # model.add(Dropout(0.5))
    # layer6
    model.add(Dense(2, name="preds", activation=None))
    model.add(out)
    # print(model.summary())
    return model


def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    """Inception module"""
    # 1x1 conv
    conv1 = Conv2D(f1, (1,1), data_format = 'channels_first', padding='same', activation=None)(layer_in)
    conv1 = keras.activations.gelu(conv1)
    # 3x3 conv
    conv3 = Conv2D(f2_in, (1,1),data_format = 'channels_first', padding='same', activation=None)(layer_in)
    conv3 = keras.activations.gelu(conv3)
    conv3 = Conv2D(f2_out, (3,3), data_format = 'channels_first', dilation_rate = (2, 2),padding='same', activation=None)(conv3)
    conv3 = keras.activations.gelu(conv3)
    # 5x5 conv
    conv5 = Conv2D(f3_in, (1,1),data_format = 'channels_first', padding='same', activation=None)(layer_in)
    conv5 = keras.activations.gelu(conv5)
    conv5 = Conv2D(f3_out, (5,5),data_format = 'channels_first', dilation_rate = (3, 3), padding='same', activation=None)(conv5)
    conv5 = keras.activations.gelu(conv5)
    # 3x3 max pooling
    pool = MaxPooling2D((3,3), data_format = 'channels_first', strides=(1,1), padding='same')(layer_in)
    pool = Conv2D(f4_out, (1,1),data_format = 'channels_first', padding='same', activation=None)(pool)
    pool = keras.activations.gelu(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=1)
    return layer_out


def EEGnet_Inception(windows_size, n_channel_input):
    # Defining some hyperparameters for ConvMixer-1536/20
    input_shape = (1, n_channel_input, windows_size)
    factor_time = 1

    # Input and patch embedding layer
    input = Input(input_shape)

    # add inception block 1
    x = inception_module(input, 16 // factor_time, 20 // factor_time, 24 // factor_time, 4 // factor_time, 10 // factor_time, 10 // factor_time)
    # add inception block 2
    # x = inception_module(x, 64// factor_time, 64// factor_time, 98// factor_time, 16// factor_time, 48// factor_time, 32// factor_time)
    # add inception block 3
    x = inception_module(x, 32 // factor_time, 50// factor_time, 52 // factor_time, 8 // factor_time, 30 // factor_time, 32 // factor_time)

    # Classification head
    x = GlobalAveragePooling2D(data_format='channels_first',)(x)
    x = Flatten()(x)
    x = Dense(128, activation=None)(x)
    x = LeakyReLU(alpha=0.3)(x)
    outputs = Dense(units=2, activation="softmax")(x)

    eegnet_inception = keras.Model(inputs=input, outputs=outputs)

    return eegnet_inception

def vanilliaEEG2Code2(windows_size,n_channel_input):
    # construct sequential model
    model = Sequential()
    # permute input so that it is as in EEG Net paper
    model.add(InputLayer(input_shape=(1, n_channel_input, windows_size)))
    # layer1
    model.add(Conv2D(16, kernel_size=(n_channel_input, 1), padding='valid', strides=(1, 1), data_format='channels_first', activation='relu'))
    model.add(BatchNormalization(axis=1, scale=False, center=False))
    #model.add(Permute((2,1,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='same'))
    # layer2
    model.add(Conv2D(8,kernel_size=(1, 64),data_format='channels_first',padding='same'))
    model.add(BatchNormalization(axis=1,scale=False, center=False))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='same'))
    model.add(Dropout(0.5))
    # layer3
    model.add(Conv2D(4,kernel_size=(5, 5),data_format='channels_first',padding='same'))
    model.add(BatchNormalization(axis=1,scale=False,center=False))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first',padding='same'))
    model.add(Dropout(0.5))
    # layer4
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    # layer5
    model.add(Dense(2, activation='softmax'))
    #print(model.summary())
    return model

def vanilliaEEG2Code(windows_size,n_channel_input):
    # construct sequential model
    model = Sequential()
    # permute input so that it is as in EEG Net paper
    model.add(InputLayer(input_shape=(1, n_channel_input, windows_size)))
    # layer1
    model.add(Conv2D(16, kernel_size=(n_channel_input, 1), padding='valid', strides=(1, 1), data_format='channels_first', activation='relu'))
    model.add(BatchNormalization(axis=1, scale=False, center=False))
    #model.add(Permute((2,1,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='same'))
    # layer2
    model.add(Conv2D(8,kernel_size=(1, 64),data_format='channels_first',padding='same'))
    model.add(BatchNormalization(axis=1,scale=False, center=False))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='same'))
    model.add(Dropout(0.5))
    # layer3
    model.add(Conv2D(4,kernel_size=(5, 5),data_format='channels_first',padding='same'))
    model.add(BatchNormalization(axis=1,scale=False,center=False))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first',padding='same'))
    model.add(Dropout(0.5))
    # layer4
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    # layer5
    model.add(Dense(2, activation='softmax'))
    #print(model.summary())
    return model



if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model
    #import visualkeras

    clf = basearchi_patchembeddingdepthwise(125, 30)
    clf.summary()
    #visualkeras.layered_view(clf, to_file='models_archi/EEGnet_patchembeddingdepthwise_viz.png', legend=True, draw_volume=True)
    plot_model(clf, to_file='models_archi/EEGnet_patchembeddingdepthwise.png', show_shapes=True, show_layer_names=True)
