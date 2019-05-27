from .backend import resnet50, se_resnet50
from tensorflow.python.keras.models import Model, Input
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Flatten, Dense, AveragePooling2D
from tensorflow.python.keras.utils.data_utils import get_file

PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/'
FNAME_TOP = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
FNAME_NO_TOP = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
HASH_TOP = 'a7b3fe01876f51b976af0dea6bc144eb'
HASH_NO_TOP = 'a268eb855778b3df3c7506639542a6af'


def create_model(model='resnet50', input_dim=(224, 224, 3), strides=(2, 2), feature_dim=256,
                 use_bias=False, truncated=False, pooling='avg', kernel_regularizer='auto',
                 train=True, num_classes=8631):
    """Creates a keras backend model for vggface 2 dataset.

    Parameters
    ----------
    model:str
        Supported backends are 'resnet50' and 'se_resnet50'
    input_dim: tuple
        Input dimension (width, height, channels). Min size should be 160x160
    strides: int, list, tuple
        Specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for all spatial dimensions.
    feature_dim: int
        Number of classes to add model.
    use_bias: bool
        whether the layer uses bias vector
    truncated: bool
        Removes the last 3 block from model. Returned shapes are 7x7
    pooling: str, None
        Optional pooling mode for feature extraction.
            - `None` means that the output of the model will be the 4D tensor output
            of the last convolutional layer. NOT IMPLEMENTED
            - `avg` means that global average pooling will be applied to the output of the
            last convolutional layer, and thus the output of the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
    kernel_regularizer: str, function
        Define a kernel regularizer to pass conv layers.
        Default 'auto' - l2(1e-4) for resnet50 and None for se_resnet50.
    train: bool
        Whether to train the data or not. If model weights are present, it can be loaded
        after model is created.Set this to false only if you want to map the model
        to desired
    num_classes: int
        Number of classes(as different person in VGGFace 2 case). Only valid when train is True

    Returns
    -------
        model: Model
            A keras Model instance
    """

    if model not in ['resnet50', 'se_resnet50']:
        raise ('The model selection is not supported for now. Please select a valid model. Options are \
              resnet50 or se_resnet50..')

    if (input_dim[0] < 160) or (input_dim[1] < 160):
        raise ('Please give an image with greater size than 160x160. Make sure the channel is the \
                last one..')

    inputs = Input(shape=input_dim, name='base_input')

    if model is 'resnet50':
        regularizer = l2(1e-4) if kernel_regularizer is 'auto' else kernel_regularizer
        x = resnet50(inputs=inputs,
                     strides=strides,
                     use_bias=use_bias,
                     truncated=truncated,
                     pooling=pooling,
                     kernel_regularizer=regularizer)

    else:
        regularizer = None if kernel_regularizer is 'auto' else kernel_regularizer
        x = se_resnet50(inputs=inputs,
                        strides=strides,
                        use_bias=use_bias,
                        truncated=truncated,
                        pooling=pooling,
                        kernel_regularizer=regularizer)

    # Map outputs to a FC layer.
    # This part can differ from other implementations!
    # Another implementation without global average pooling
    #     x = AveragePooling2D((7, 7), name='avg_pool')(x)
    #     x = Flatten()(x)
    #     x = Dense(512, activation='relu', name='dim_proj')(x)
    if pooling is None:
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    y = Dense(feature_dim, activation='relu', name='dim_proj')(x)

    if train:
        y = Dense(num_classes, activation='softmax', use_bias=use_bias,
                  kernel_regularizer=regularizer, name='classifier_low_dim')(y)

    model = Model(inputs=inputs, outputs=y)
    return model


def load_weights(model, fname=FNAME_NO_TOP, path=PATH, md5_hash=HASH_NO_TOP, by_name=False):
    """ Loads model weights from a remote server.

    PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/ + fname

    FNAME_TOP = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'

    FNAME_NO_TOP = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    HASH_TOP = 'a7b3fe01876f51b976af0dea6bc144eb'

    HASH_NO_TOP = 'a268eb855778b3df3c7506639542a6af'

    Parameters
    ----------
    model: Model
        A tf.keras Model instance to load weights from.
    fname: str
        Name of the file. Default is 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    path: str
        URL path to the file. Can be also a local file. Default is PATH variable. Data will be
        download from path+fname address.
    md5_hash: str
        md5 hash string of the file
    by_name: bool
        Whether to load weights by name or by topological order. Only topological loading is
        supported for weight files in TensorFlow format.

    """
    weights = get_file(fname=fname, origin=path+fname, cache_subdir='models', file_hash=md5_hash)
    model.load_weights(weights, by_name=by_name)

