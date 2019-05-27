from pandas import read_csv


def read_vgg_data(fname, sep=' ', header=None, only_labels=False):
    """ Returns training/testing data file names.

    Parameters
    ----------
    fname:str
        File name of the csv file containing train/test file names
    sep: str
        Seperator used in pandas read csv function.
    header: str
        Header in csv file
    only_labels: bool
        Whether to return labels only. Default is false

    Returns
    -------
    file_names, labels or labels:
        File names and labels
    """

    df = read_csv(fname, sep=sep, header=header)
    df = df[0].str.extract(r'(?P<ImageID>n(?P<ClassID>\d+)/\d+_(?P<FaceID>\d+).jpg)')
    df['ClassID'] = df['ClassID'].astype('int')
    df['FaceID'] = df['FaceID'].astype('int')
    if only_labels:
        return df.ClassID.values
    return df.ImageID.values, df.ClassID.values
