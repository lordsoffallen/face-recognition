from tensorflow.python.ops.math_ops import cast
from tensorflow.python.ops.gen_io_ops import read_file
from tensorflow.python.ops.gen_array_ops import reverse_v2
from tensorflow.python.ops.gen_image_ops import decode_jpeg, decode_png
from tensorflow.python.ops.image_ops_impl import resize_images
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.data import Dataset
import numpy as np


mean = (91.4953, 103.8827, 131.0912)


def _parser(file_name, label=None, shape=(224, 224), extension='jpg'):
    """ Image data parser function. It takes a path to an image,
    reads it, resizes to shape parameter and finally scaled input
    to (0, 1) interval.

    Parameters
    ----------
    file_name: str
        File name of the image
    label: int, None
        Image labels. If None, no label will be returned
    shape: tuple, list
        Resize shape parameter. Default is (224, 224)
    extension: str
        Image file extension. Supported png and jpeg for now.
        Default value is jpg

    Returns
    -------
    image_norm or image_norm, label: Tensor
        Processed image with or without labels
    """

    image_string = read_file(file_name)
    if extension is 'jpg':
        image_decoded = decode_jpeg(image_string, channels=3)
    else:
        image_decoded = decode_png(image_string, channels=3)
    image_resized = resize_images(image_decoded, size=shape)

    # Convert rgb to bgr
    image_bgr = reverse_v2(cast(image_resized, 'float32'), axis=[-1])
    image_norm = image_bgr - constant([91.4953, 103.8827, 131.0912], 'float32')
    if label is None:
        return image_norm
    return image_norm, label


def get_data(file_names, labels=None, shape=(224, 224), train=True, cores=4,
             batch_size=32, buffer_size=1, extension='jpg', gpu=False):
    """ Prepares the ETL for the model. Reads, maps and returns the data.

    Parameters
    ----------
    file_names: list of str
        A list of strings of image file names
    labels: list of int, None
        A list of ints contain image labels
    shape: tuple of int, list of int
        Output image shapes of the read data.
    train: bool
        Data shuffled and repeated when train is True. Otherwise
        map, batch, prefetch. Default is true
    cores: int
        Number of parallel jobs for mapping function.
    batch_size: int
        Number of batch size
    buffer_size: int
        A prefetch buffer size to speed up to parallelism
    extension: str
        Image file extension. Supported png and jpeg for now.
        Default value is jpg
    gpu: bool
        If true then it adds support for dataset prefetching to speed up
        to data pipeline.

    Returns
    -------
    dataset: Dataset
        Tensorflow Dataset
    """

    if train:
        dataset = Dataset.from_tensor_slices((file_names, labels))
        func = lambda x, y: _parser(file_name=x, label=y, shape=shape, extension=extension)
        dataset = (dataset.shuffle(buffer_size=1000)
                          .repeat(len(file_names))
                          .map(map_func=func, num_parallel_calls=cores)
                          .batch(batch_size))
    else:
        dataset = Dataset.from_tensor_slices(file_names)
        func = lambda x: _parser(file_name=x, shape=shape, extension=extension)
        dataset = (dataset.map(map_func=func, num_parallel_calls=cores)
                          .batch(batch_size))

    if gpu:
        dataset = dataset.prefetch(buffer_size)

    return dataset


def check_and_load_weights(model, weights, by_name=True):
    """ Check and load weights from h5 file.
    If model weights cannot be loaded properly, raises Error

    Parameters
    ----------
    model: Model
        Tensorflow Keras Model instance
    weights: str
        Weights path
    by_name: bool
        If `by_name` is False weights are loaded based on the network's
        topology. This means the architecture should be the same as when the weights
        were saved.  Note that layers that don't have weights are not taken into
        account in the topological ordering, so adding or removing layers is fine as
        long as they don't have weights.
        If `by_name` is True, weights are loaded into layers only if they share the
        same name.
        Only topological loading (`by_name=False`) is supported when loading weights
        from the TensorFlow format. Note that topological loading differs slightly
        between TensorFlow and HDF5 formats for user-defined classes inheriting from
    `   tf.keras.Model`: HDF5 loads based on a flattened list of weights, while the
        TensorFlow format loads based on the object-local names of attributes to
        which layers are assigned in the `Model`'s constructor.

    """
    # store weights before loading pre-trained weights
    preloaded_layers = model.layers.copy()
    preloaded_weights = [pre.get_weights() for pre in preloaded_layers]

    # load pre-trained weights
    model.load_weights(weights, by_name=by_name)

    # compare previews weights vs loaded weights
    for layer, pre in zip(model.layers, preloaded_weights):
        _weights = layer.get_weights()
        if _weights:
            if np.array_equal(_weights, pre):
                raise ('{} layer weights cannot be loaded. Make \
                 sure model names are matching or change by_name to False'.format(layer.name))
