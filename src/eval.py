from tensorflow.python.keras import Model
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from .utils import get_data
import numpy as np


def image_embeddings(file_names, model, shape=(224, 224), cores=4,
                     batch_size=32, buffer_size=1, flat=True, extension='jpg'):
    """ Retrieve image embeddings from the model

    Parameters
    ----------
    file_names: list of str
        A list of strings of image file names
    model: Model
        Keras model instace to apply predict generator
    shape: tuple, list
        Output image shapes of the read data.
    cores: int
        Number of parallel jobs for mapping function.
    batch_size: int
        Number of batch size
    buffer_size: int
        A prefetch buffer size to speed up to parallelism
    flat: bool
        If true, return a flattenned output
    extension: str
        Image file extension. Supported png and jpeg for now.
        Default value is jpg

    Returns
    -------
    embeddings: numpy.ndarray
        Returns image embeddings from model. Expected return size
        is number of images x embedding dimension
    """

    x = get_data(file_names, shape=shape, train=False, cores=cores,
                 batch_size=batch_size, buffer_size=buffer_size, extension=extension)

    # predict the image embeddings.
    predictions = model.predict(x, steps=len(file_names) // batch_size, verbose=1)
    if flat:
        return predictions.flatten()
    return predictions


def print_scores(y_true, y_pred):
    print('ACC Score : ', accuracy_score(y_true, y_pred))
    print('F1 Score with macro avg : ', f1_score(y_true, y_pred, average='macro'))
    print('F1 Scores with micro avg: ', f1_score(y_true, y_pred, average='micro'))
    print('F1 Scores with micro weighted: ', f1_score(y_true, y_pred, average='weighted'))
    print('Precision with macro avg : ', precision_score(y_true, y_pred, average='macro'))
    print('Precision with micro avg : ', precision_score(y_true, y_pred, average='micro'))
    print('Precision with micro weighted : ', precision_score(y_true, y_pred, average='weighted'))


def fit_classifier(X_train, y_train, X_test, y_test, model='logistic',
                   C=1.0, kernel='linear', verbose=1, report=False):
    """ Fits a classifier on image embeddings. SVM and Logistic Regression
    performs well on FERET dataset.

    Parameters
    ----------
    X_train: np.ndarray
        Training embeddings for classifier
    y_train: np.ndarray
        Train data labels
    X_test: np.ndarray
        Test embeddings for classifier
    y_test: np.ndarray
        Test data labels
    model: str
        Options are svm or logistic regression
    C: float
        C parameter for logistic regression and svm
    kernel: str
        Only valid for svm model. Defines kernel type. Default is linear
        Options are rbf and linear.
    verbose: int
        Whether to print process or not. 0 means print nothing.
    report: bool
        Whether to print classification report or not. Beware that this
        is a long output as there are a lot of classes. It may be hard to interpret.

    Returns
    -------
    estimator:
        Returns the trained model estimator.
    """

    if model is 'logistic':
        estimator = LogisticRegression(solver='lbfgs', n_jobs=-1, verbose=verbose, max_iter=500, C=C)
    else:
        estimator = SVC(verbose=verbose, kernel=kernel, C=C)

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    print_scores(y_test, y_pred)
    if report:
        print(classification_report(y_test, y_pred))
    return estimator

