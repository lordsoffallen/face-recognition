from sklearn.metrics import roc_auc_score
from src import create_model, check_and_load_weights
from src import image_embeddings
from src import ijb_utils as ijb
import os.path as path
import numpy as np
import argparse


WEIGHTS = 'model/resnet50_softmax_dim512/weights.h5'
IJBB_DATA_PATH = path.join('..', 'IJB_release', 'IJBB', 'loose_crop')
IJBC_DATA_PATH = path.join('..', 'IJB_release', 'IJBC', 'loose_crop')

IJBB_PREDICTION_PATH = path.join('predictions','ijbb_predictions.npy')
IJBC_PREDICTION_PATH = path.join('predictions','ijbc_predictions.npy')

IJBB_EMBEDDING_PATH = path.join('embeddings', 'ijbb_embeddings.npy')
IJBC_EMBEDDING_PATH = path.join('embeddings', 'ijbc_embeddings.npy')

IJBB_TEMPLATE_PATH = path.join('templates', 'ijbb_templates.npy')
IJBC_TEMPLATE_PATH = path.join('templates', 'ijbc_templates.npy')

# 1. Train more models to add more options to feature dimension
# 2. Add more backend options
# 4. Train the model on se_resnet50


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet50', choices=['resnet50', 'se_resnet50'], type=str)
parser.add_argument('--platform', default='ijbc', choices=['ijbb', 'ijbc'], type=str)
parser.add_argument('--batch_size', default=25, type=int)
parser.add_argument('--feature_dim', default=512, choices=[512], type=int)
parser.add_argument('--cores', default=8, type=int)

args = parser.parse_args()


if __name__ == '__main__':
    if args.platform is 'ijbb':
        # Get face image data paths, templates and medias
        face_paths, templates, medias = ijb.get_meta_info(IJBB_DATA_PATH)

        # Get y labels and templates(p1, p2)
        y_true, p1, p2 = ijb.get_template_pair_label()

        # Check if it is pre-computed
        if path.isfile(IJBB_EMBEDDING_PATH):
            face_features = np.load(IJBB_EMBEDDING_PATH)
        else:
            resnet50 = create_model(model=args.model, input_dim=(224, 224, 3),
                                    pooling=None, feature_dim=args.feature_dim, train=False)
            check_and_load_weights(resnet50, WEIGHTS)

            # Retrieve predictions from the model
            face_features = image_embeddings(file_names=face_paths, model=resnet50,
                                             batch_size=args.batch_size, cores=args.cores)
            np.save(IJBB_EMBEDDING_PATH, face_features)

        # Check if it is pre-computed
        if path.isfile(IJBB_TEMPLATE_PATH):
            template_features = np.load(IJBB_TEMPLATE_PATH)
            template_features = template_features.astype(np.int32)
        else:
            # Retrieve template embeddings
            template_features = ijb.template_embeddings(templates, medias, face_features, feature_dim=args.feature_dim)
            np.save(IJBB_TEMPLATE_PATH, template_features)

        # Check if it is pre-computed
        if path.isfile(IJBB_PREDICTION_PATH):
            y_pred = np.load(IJBB_PREDICTION_PATH)
        else:
            # Retrieve prediction scores
            y_pred = ijb.verification(unique_templates=np.unique(templates),
                                      template_features=template_features,
                                      p1=p1, p2=p2, batchsize=args.batch_size)
            np.save(IJBB_PREDICTION_PATH, y_pred)

        auc = roc_auc_score(y_true, y_pred)
        ijb.compute_roc(y_true, y_pred)

    else:
        # Get face image data paths, templates and medias
        face_paths, templates, medias = ijb.get_meta_info(IJBC_DATA_PATH, platform='ijbc')

        # Get y labels and templates(p1, p2)
        y_true, p1, p2 = ijb.get_template_pair_label(platform='ijbc')

        # Check if it is pre-computed
        if path.isfile(IJBC_EMBEDDING_PATH):
            face_features = np.load(IJBC_EMBEDDING_PATH)
        else:
            resnet50 = create_model(model=args.model, input_dim=(224, 224, 3),
                                    pooling=None, feature_dim=args.feature_dim, train=False)
            check_and_load_weights(resnet50, WEIGHTS)

            # Retrieve predictions from the model
            face_features = image_embeddings(file_names=face_paths, model=resnet50, flat=False,
                                             batch_size=args.batch_size, cores=args.cores)
            np.save(IJBC_EMBEDDING_PATH, face_features)

        # Check if it is pre-computed
        if path.isfile(IJBC_TEMPLATE_PATH):
            template_features = np.load(IJBC_TEMPLATE_PATH)
            # template_features = template_features.astype(np.int32)
        else:
            # Retrieve template embeddings
            template_features = ijb.template_embeddings(templates, medias, face_features, feature_dim=args.feature_dim)
            np.save(IJBC_TEMPLATE_PATH, template_features)

        # Check if it is pre-computed
        if path.isfile(IJBC_PREDICTION_PATH):
            y_pred = np.load(IJBC_PREDICTION_PATH)
        else:
            # Retrieve prediction scores
            y_pred = ijb.verification(unique_templates=np.unique(templates),
                                      template_features=template_features,
                                      p1=p1, p2=p2, batchsize=args.batch_size)
            np.save(IJBC_PREDICTION_PATH, y_pred)

        auc = roc_auc_score(y_true, y_pred)
        ijb.compute_roc(y_true, y_pred, roc_csv='TAR_AT_FAR_IJBC.csv')




