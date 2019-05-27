from tensorflow.python.keras import layers
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import (Conv2D, Activation, BatchNormalization,
                                            MaxPooling2D, AveragePooling2D, Dense,
                                            GlobalAveragePooling2D, GlobalMaxPooling2D,
                                            Multiply, Reshape)


def identity_block(input_tensor, kernel_size, filters, stage, block, 
                   kernel_regularizer=None, use_bias=False, squeeze_and_excitation=False):
    """The identity block is the block that has no conv layer at shortcut.

    Parameters
    ----------
    input_tensor: Tensor
        input tensor
    kernel_size: int, tuple
        default 3, the kernel size of middle conv layer at main path
    filters: list
        list of integers, the filters of 3 conv layer at main path
    stage: int
        integer, current stage label, used for generating layer names
    block: int
        integer, current block label, used for generating layer names
    kernel_regularizer: function
        Define a kernel regularizer. Possibly l2(1e-4)
    use_bias:
        boolean, whether the layer uses bias vector
    squeeze_and_excitation:
        whether to add squeeze_and_excitation part
        to the block or not.

    Returns
    -------
        x: Tensor
            Layer Output Tensor
    """

    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'conv' + str(stage) + '_' + str(block)
    bn_name_base = 'conv' + str(stage) + '_' + str(block)

    x = Conv2D(filters1, (1, 1),
               use_bias=use_bias,
               kernel_regularizer=kernel_regularizer,
               name=conv_name_base + '_1x1_reduce')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '_1x1_reduce/bn')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', 
               use_bias=use_bias,
               kernel_regularizer=kernel_regularizer,
               name=conv_name_base + '_3x3')(x)
    x = BatchNormalization(axis=bn_axis,  name=bn_name_base + '_3x3/bn')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
               use_bias=use_bias,
               kernel_regularizer=kernel_regularizer,
               name=conv_name_base + '_1x1_increase')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '_1x1_increase/bn')(x)

    if squeeze_and_excitation:
        se = GlobalAveragePooling2D(name='pool' + str(block) + '_gap')(x)
        se = Dense(filters3 // 16, activation='relu', name='fc' + str(block) + '_sqz')(se)
        se = Dense(filters3, activation='sigmoid', name='fc' + str(block) + '_exc')(se)
        se = Reshape([1, 1, filters3])(se)
        x = Multiply(name='scale' + str(block))([x, se])

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),
               kernel_regularizer=None, use_bias=False, squeeze_and_excitation=False):
    """The identity block is the block that has no conv layer at shortcut.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well

    Parameters
    ----------
    input_tensor: Tensor
        input tensor
    kernel_size: int, tuple
        default 3, the kernel size of middle conv layer at main path
    filters: list
        list of integers, the filters of 3 conv layer at main path
    stage: int
        integer, current stage label, used for generating layer names
    block: int
        integer, current block label, used for generating layer names
    strides: int, tuple, list
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for all spatial dimensions.
    kernel_regularizer: function
        Define a kernel regularizer. Possibly l2(1e-4)
    use_bias: bool
        boolean, whether the layer uses bias vector
    squeeze_and_excitation:
        whether to add squeeze_and_excitation part
        to the block or not.

    Returns
    -------
        x: Tensor
            Tensor object
    """

    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'conv' + str(stage) + '_' + str(block)
    bn_name_base = 'conv' + str(stage) + '_' + str(block)

    x = Conv2D(filters1, (1, 1), 
               strides=strides,
               use_bias=use_bias,
               kernel_regularizer=kernel_regularizer,
               name=conv_name_base + '_1x1_reduce')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '_1x1_reduce/bn')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, 
               padding='same',
               use_bias=use_bias,
               kernel_regularizer=kernel_regularizer,
               name=conv_name_base + '_3x3')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '_3x3/bn')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
               use_bias=use_bias,
               kernel_regularizer=kernel_regularizer,
               name=conv_name_base + '_1x1_increase')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '_1x1_increase/bn')(x)

    if squeeze_and_excitation:
        se = GlobalAveragePooling2D(name='pool' + str(block) + '_gap')(x)
        se = Dense(filters3 // 16, activation='relu', name = 'fc' + str(block) + '_sqz')(se)
        se = Dense(filters3, activation='sigmoid', name = 'fc' + str(block) + '_exc')(se)
        se = Reshape([1, 1, filters3])(se)
        x = Multiply(name='scale' + str(block))([x, se])

    shortcut = Conv2D(filters3, (1, 1), 
                      strides=strides,
                      use_bias=use_bias,
                      kernel_regularizer=kernel_regularizer,
                      name=conv_name_base + '_1x1_proj')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '_1x1_proj/bn')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50(inputs, strides=(2, 2), use_bias=False, truncated=False, pooling=None,
             kernel_regularizer=l2(1e-4)):
    """ResNet Backend

    Parameters
    ----------
    inputs: Tensor
        input tensors
    strides: int, tuple, list
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for all spatial dimensions.
    use_bias: bool
        boolean, whether the layer uses bias vector
    truncated: bool
         Removes the last 3 block from model. Returned shapes are 7x7
    pooling: str, None
        Optional pooling mode for feature extraction:
        - `None` means that the output of the model will be the 4D tensor output
        of the last convolutional layer.
        - `avg` means that global average pooling will be applied to the output of the
        last convolutional layer, and thus the output of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
    kernel_regularizer: function
        Define a kernel regularizer to pass conv layers. Default l2(1e-4)

    Returns
    -------
        x: Tensor
            Tensor containing embedded features
    """

    bn_axis = 3

    # input sizes 224 x 224 x 3
    x = Conv2D(64, (7, 7), strides=(2, 2),
               use_bias=use_bias,
               kernel_regularizer=kernel_regularizer,
               padding='same',
               name='conv1/7x7_s2')(inputs)

    # input sizes 112 x 112 x 64
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # input sizes 56 x 56
    x = conv_block(x, (3, 3), filters=[64, 64, 256], stage=2, block=1, strides=(1, 1), 
                   kernel_regularizer=kernel_regularizer, use_bias=use_bias)
    x = identity_block(x, (3, 3), filters=[64, 64, 256], stage=2, block=2,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias)
    x = identity_block(x, (3, 3), filters=[64, 64, 256], stage=2, block=3,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias)

    # input sizes 28 x 28
    x = conv_block(x, (3, 3), filters=[128, 128, 512], stage=3, block=1, strides=strides, 
                   kernel_regularizer=kernel_regularizer, use_bias=use_bias)
    x = identity_block(x, (3, 3), filters=[128, 128, 512], stage=3, block=2,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias)
    x = identity_block(x, (3, 3), filters=[128, 128, 512], stage=3, block=3,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias)
    x = identity_block(x, (3, 3), filters=[128, 128, 512], stage=3, block=4,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias)

    # input sizes 14 x 14
    x = conv_block(x, (3, 3), filters=[256, 256, 1024], stage=4, block=1, strides=strides, 
                   kernel_regularizer=kernel_regularizer, use_bias=use_bias)
    x = identity_block(x, (3, 3), filters=[256, 256, 1024], stage=4, block=2,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias)
    x = identity_block(x, (3, 3), filters=[256, 256, 1024], stage=4, block=3,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias)
    x = identity_block(x, (3, 3), filters=[256, 256, 1024], stage=4, block=4,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias)
    x = identity_block(x, (3, 3), filters=[256, 256, 1024], stage=4, block=5,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias)
    x = identity_block(x, (3, 3), filters=[256, 256, 1024], stage=4, block=6,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias)

    # truncated
    if truncated:
        return x

    # input sizes 7 x 7
    x = conv_block(x, (3, 3), filters=[512, 512, 2048], stage=5, block=1, strides=strides,
                   kernel_regularizer=kernel_regularizer, use_bias=use_bias)
    x = identity_block(x, (3, 3), filters=[512, 512, 2048], stage=5, block=2,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias)
    x = identity_block(x, (3, 3), filters=[512, 512, 2048], stage=5, block=3,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias)
    
    if pooling is 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling is 'max':
        x = GlobalMaxPooling2D()(x)

    return x


def se_resnet50(inputs, strides=(2, 2), use_bias=False, truncated=False, pooling=None,
                kernel_regularizer=None):
    """ Squeeze-and-Excitation ResNet50 Backend

    Parameters
    ----------
    inputs: Tensor
        input tensors
    strides: int, tuple, list
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for all spatial dimensions.
    use_bias: bool
        boolean, whether the layer uses bias vector
    truncated: bool
         Removes the last 3 block from model. Returned shapes are 7x7
    pooling: str, None
        Optional pooling mode for feature extraction:
        - `None` means that the output of the model will be the 4D tensor output
        of the last convolutional layer.
        - `avg` means that global average pooling will be applied to the output of the
        last convolutional layer, and thus the output of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
    kernel_regularizer: function
        Define a kernel regularizer to pass conv layers. Default None

    Returns
    -------
        x: Tensor
            Tensor containing embedded features
    """

    bn_axis = 3

    # input sizes 224 x 224 x 3
    x = Conv2D(64, (7, 7), strides=(2, 2),
               use_bias=use_bias,
               kernel_regularizer=kernel_regularizer,
               padding='same',
               name='conv1/7x7_s2')(inputs)

    # input sizes 112 x 112 x 64
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # input sizes 56 x 56
    x = conv_block(x, (3, 3), filters=[64, 64, 256], stage=2, block=1, strides=(1, 1), 
                   kernel_regularizer=kernel_regularizer, use_bias=use_bias, 
                   squeeze_and_excitation=True)
    x = identity_block(x, (3, 3), filters=[64, 64, 256], stage=2, block=2, 
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                       squeeze_and_excitation=True)
    x = identity_block(x, (3, 3), filters=[64, 64, 256], stage=2, block=3,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                       squeeze_and_excitation=True)

    # input sizes 28 x 28
    x = conv_block(x, (3, 3), filters=[128, 128, 512], stage=3, block=1, strides=strides, 
                   kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                   squeeze_and_excitation=True)
    x = identity_block(x, (3, 3), filters=[128, 128, 512], stage=3, block=2,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                       squeeze_and_excitation=True)
    x = identity_block(x, (3, 3), filters=[128, 128, 512], stage=3, block=3,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                       squeeze_and_excitation=True)
    x = identity_block(x, (3, 3), filters=[128, 128, 512], stage=3, block=4,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                       squeeze_and_excitation=True)

    # input sizes 14 x 14
    x = conv_block(x, (3, 3), filters=[256, 256, 1024], stage=4, block=1, strides=strides, 
                   kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                   squeeze_and_excitation=True)
    x = identity_block(x, (3, 3), filters=[256, 256, 1024], stage=4, block=2,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                       squeeze_and_excitation=True)
    x = identity_block(x, (3, 3), filters=[256, 256, 1024], stage=4, block=3,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                       squeeze_and_excitation=True)
    x = identity_block(x, (3, 3), filters=[256, 256, 1024], stage=4, block=4,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                       squeeze_and_excitation=True)
    x = identity_block(x, (3, 3), filters=[256, 256, 1024], stage=4, block=5,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                       squeeze_and_excitation=True)
    x = identity_block(x, (3, 3), filters=[256, 256, 1024], stage=4, block=6,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                       squeeze_and_excitation=True)

    if truncated:
        return x

    # input sizes 7 x 7
    x = conv_block(x, (3, 3), filters=[512, 512, 2048], stage=5, block=1, strides=strides,
                   kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                   squeeze_and_excitation=True)
    x = identity_block(x, (3, 3), filters=[512, 512, 2048], stage=5, block=2,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                       squeeze_and_excitation=True)
    x = identity_block(x, (3, 3), filters=[512, 512, 2048], stage=5, block=3,
                       kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                       squeeze_and_excitation=True)
    
    if pooling is 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling is 'max':
        x = GlobalMaxPooling2D()(x)

    return x

