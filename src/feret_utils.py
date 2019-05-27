from pandas import DataFrame
from glob import glob
import numpy as np
import os.path as path


def read_feret_data(imgs_folder='feret', extension='jpg', train=True, only_labels=False):
    """ Read FERET image file names and extract labels from it.

    Parameters
    ----------
    imgs_folder: str
        Path, where the images are located. This should be the name of the folder
        and there should be 2 different folders inside this folder named as
        train and test.
    extension: str
        Image file extension. Default is jpg
    train: bool
        Whether to select train or test files. If false, it means choose
        test files from imgs_folder/test/FILES...
    only_labels: bool
        If this is True, then only label values will returned as a numpy array

    Returns
    -------
    df or y_true: DataFrame or np.ndarray
        A dataframe contains names of the files, personids and positions.
        Column names are : Names, PersonID, Position.

        If only_labels was set to True then just a ndarray for image labels
    """

    if train:
        files = glob(path.join(imgs_folder, 'train', '*.{}'.format(extension)))
        df = DataFrame(files, columns=['Names'])
        df = df.Names.str\
            .extract(r'(?P<Names>{}\\train\\(?P<PersonID>\d+)_\d+_(?P<Position>\w\w)_?(?P<Optional>[a-c]?).{})'
                     .format(imgs_folder, extension))
    else:
        files = glob(path.join(imgs_folder, 'test', '*.{}'.format(extension)))
        df = DataFrame(files, columns=['Names'])
        df = df.Names.str\
            .extract(r'(?P<Names>{}\\test\\(?P<PersonID>\d+)_\d+_(?P<Position>\w\w)_?(?P<Optional>[a-c]?).{})'
                     .format(imgs_folder, extension))
    df['PersonID'] = df['PersonID'].astype('int')

    if only_labels:
        return df.PersonID.values

    # Map empty strings to NaN
    df['Optional'] = df['Optional'].where(df.Optional != '', np.nan)
    return df
