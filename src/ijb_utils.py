from scipy import interpolate
from sklearn.metrics import roc_curve
from pandas import read_csv
import numpy as np
import os.path as path


def get_meta_info(folder_path, platform='ijbb', sep=' ', header=None, dtype=None):
    """ Reads meta data file and imports file names. (E.g 1.jpg, 2.jpg...)


    Parameters
    ----------
    folder_path: str
        Location of the face pictures. Note that this should be a folder
        not a file path. File names are inferred from the csv file located
        in meta/$PLATFORM_face_tid_mid.csv file.
    platform: str
        IJB platform. Options are ijbb or ijbc
    sep: str
        Separater for the files. Default is space. It is passed to read_csv
        function from pandas
    header: str
        Header for the file. Default file has no header.
    dtype: dict
        Dtypes for each column. {COL_NAME: type(preferably from numpy)}.
        It is passed to read_csv function from pandas. For more details
        refer to their documentations.

        Default values are :
        {0: np.str, 1: np.int, 2: np.int}

    Returns
    -------
    face_paths, templates, medias:
        Returns a tuple consists of face_paths, templates and medias.

    """

    # Data types for columns
    dtypes = {0: np.str, 1: np.int, 2: np.int} if dtype is None else dtype

    if platform not in ['ijbb', 'ijbc']:
        raise ('IJB supported platforms are IJBB and IJBC.')

    # Take Transpose to retrive column based values and unpack it to 3 variables
    faces, templates, medias = read_csv('meta/{}_face_tid_mid.csv'.format(platform),
                                        sep=sep, header=header, dtype=dtypes).T.values
    face_paths = np.array([path.join(folder_path, f) for f in faces])
    templates = templates.astype(np.int32)
    medias = medias.astype(np.int32)
    return face_paths, templates, medias


def get_template_pair_label(platform='ijbb', sep=' ', header=None, dtype=None):
    """ Loads meta information for template-to-template verification.
    tid --> template id,  label --> 1/0
    format: tid_1 tid_2 label

    Parameters
    ----------
    platform: str
        IJB platform. Options are ijbb or ijbc
    sep: str
        Separater for the files. Default is space. It is passed to read_csv
        function from pandas
    header: str
        Header for the file. Default file has no header.
    dtype:
        Dtypes for each column. {COL_NAME: type(preferably from numpy)}.
        It is passed to read_csv function from pandas. For more details
        refer to their documentations.

        Default values are :
        {0: np.int, 1: np.int, 2: np.int}

    Returns
    -------
    face_paths, templates, medias:
        Returns a tuple consists of face_paths, templates and medias.

    """

    # Data types for columns
    dtypes = {0: np.int, 1: np.int, 2: np.int} if dtype is None else dtype

    if platform not in ['ijbb', 'ijbc']:
        raise ('IJB supported platforms are IJBB and IJBC.')

    p1, p2, y = read_csv('meta/{}_template_pair_label.csv'.format(platform),
                         sep=sep, header=header, dtype=dtypes).T.values

    return y, p1, p2


def template_embeddings(templates, medias, img_features, feature_dim=512):
    """ Face image --> l2 normalization --> Compute media encoding -->
     Compute template encoding --> Save template features.
     This function creates a generalized template features from
     image features. For instance from the model we have different pictures
     for different images but here we combine all of them into unique feature.

    Parameters
    ----------
    templates: np.ndarray
        Template ids. This is also called classes for each pictures.
        Size is (num_images)
    medias: np.ndarray
        Media id's. Size is (num_images)
    img_features: np.ndarray
        Image features from the model. Size is (num_images, feature_dim)
    feature_dim: int
        Feature dimension. Model is trained on 512 dimension. If used another parameter
        make sure that model is trained on that feature dimension.

    Returns
    -------
    template_features: np.ndarray
        Template features with the size (number of unique templates(number or people), 512)
    """

    unique_templates = np.unique(templates)
    template_features = np.empty((len(unique_templates), feature_dim))

    img_sqr = np.sqrt(np.sum(img_features ** 2, axis=-1, keepdims=True))
    img_norm_features = np.divide(img_features, img_sqr, out=np.zeros_like(img_features), where=img_sqr != 0)

    for c, template in enumerate(unique_templates):
        (ind_t,) = np.where(templates == template)
        face_norm_features = img_norm_features[ind_t]
        faces_media = medias[ind_t]
        unique_media, counts = np.unique(faces_media, return_counts=True)
        media_norm_features = []

        for media, count in zip(unique_media, counts):
            (ind_m,) = np.where(faces_media == media)
            if count < 2:
                media_norm_features.append(face_norm_features[ind_m])
            else:
                media_norm_features.append(np.sum(face_norm_features[ind_m], axis=0, keepdims=True))

        media_norm_features = np.array(media_norm_features)
        sqr = np.sqrt(np.sum(media_norm_features**2, axis=-1, keepdims=True))
        media_norm_features = np.divide(media_norm_features, sqr,
                                        out=np.zeros_like(media_norm_features), where=sqr != 0)
        template_features[c] = np.sum(media_norm_features, axis=0)
        if c % 500 == 0:
            print('Finished encoding {}/{} templates.'.format(c, len(unique_templates)))

    return template_features


def verification(unique_templates, template_features, p1, p2, batchsize=256):
    """ Compute 1:1 template verification results.
    A list of 8,010,270 comparisons is provided between S1 and
    S2 gallery templates. In total, there are 10,270 genuine
    comparisons and 8,000,000 impostor comparisons. With the
    large numbers of genuine and impostor comparison scores,
    the lower bounds of the ROC at very low FAR values can be evaluated
    (0.01% or 0.001%).

    Parameters
    ----------
    unique_templates: np.ndarray
        Unique template values. Otherwise known as unique classes or people
    template_features: np.ndarray
        Template features with a size(len(unique_templates), feature_dim)
    p1: np.ndarray
        Pair Template 1
    p2: np.ndarray
        Pair Template 2
    batchsize: int
        Batch size

    Returns
    -------
    scores:
         Returns set-to-set similarity score.
    """

    score = np.zeros((len(p1),))
    total_pairs = np.array(range(len(p1)))
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]

    for c, s in enumerate(sublists):
        id1 = np.squeeze(np.array([np.where(unique_templates == j) for j in p1[s]]))
        id2 = np.squeeze(np.array([np.where(unique_templates == j) for j in p2[s]]))

        inp1 = template_features[id1]
        inp2 = template_features[id2]

        sqr1 = np.sqrt(np.sum(np.square(inp1), axis=-1, keepdims=True))
        sqr2 = np.sqrt(np.sum(np.square(inp2), axis=-1, keepdims=True))
        v1 = np.divide(inp1, sqr1, out=np.zeros_like(inp1), where=sqr1 != 0)
        v2 = np.divide(inp2, sqr2, out=np.zeros_like(inp2), where=sqr2 != 0)

        # similarity_score
        score[s] = np.sum(v1 * v2, -1)

        if c % 500 == 0:
            print('Finished {}/{} pair verification.'.format(c, len(sublists)))

    return score


def compute_roc(y_true, y_score, roc_csv='TAR_AT_FAR.csv'):
    """ Computes the roc curve and finds TAR @ FAR values. Then
    it writes these values to the file location which was given
    as an input parameter.

    Parameters
    ----------
    y_true: numpy.array
        True label
    y_score: numpy.array
        Predicted labels
    roc_csv: str
        CSV file path to save TAR@FAR values.
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fpr_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    f_interp = interpolate.interp1d(fpr, tpr)
    tpr_at_fpr = [f_interp(x) for x in fpr_levels]
    with open('results/{}'.format(roc_csv), 'w') as file:
        for (far, tar) in zip(fpr_levels, tpr_at_fpr):
            print('TAR @ FAR = {} : {}'.format(far, tar))
            file.write('TAR @ FAR = {}: {}\n'.format(far, tar))
