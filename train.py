from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from src import create_model, get_data
from src import vggface2_utils as vgg
import os.path as path
import argparse


WEIGHTS = 'model/resnet50_softmax_dim512/weights.h5'

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='se_resnet50', choices=['resnet50', 'se_resnet50'],
                    type=str, help='Choose a specific model to train. Options are \
                                   se_resnet50 and resnet50')
parser.add_argument('--data_dir', default='meta', type=str,
                    help='Where the train/test list is located. This should be \
                         the name of the folder. As os.path will be used \
                         do not put backslash at the end.')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--feature_dim', default=512, choices=[128, 256, 512, 1024],
                    type=int, help='Feature dimension of model.')
parser.add_argument('--cores', default=8, type=int, help='Number of dedicated CPU cores')
parser.add_argument('--input_dim', default=224, type=int,
                    help='Input size for the model. It cannot be less than 160.')
parser.add_argument('--opt', default='adam', choices=['adam', 'sgd'],
                    help='Optimizer algorithm for the model')
parser.add_argument('--gpu', type=bool, default=True,
                    help='Whether the gpu is available or not \
                         If true data pipelining will be optimized in cpu')

args = parser.parse_args()


VGG_PATH = path.join('..', 'vggface2_aligned')

if __name__ == '__main__':

    # Get face image data paths, templates and medias
    file_names, labels = vgg.read_vgg_data(path.join(args.data_dir, 'train_list.csv'))
    test_file_names, test_labels = vgg.read_vgg_data(path.join(args.data_dir, 'test_list.csv'))

    file_names = [path.join(VGG_PATH, file_name) for file_name in file_names]
    test_file_names = [path.join(VGG_PATH, file_name) for file_name in test_file_names]

    dataset = get_data(file_names=file_names,
                       labels=labels, cores=args.cores,
                       batch_size=args.batch_size,
                       shape=(args.input_dim, args.input_dim, 3),
                       gpu=args.gpu)

    test_dataset = get_data(file_names=test_file_names,
                            labels=test_labels, cores=args.cores,
                            batch_size=args.batch_size,
                            shape=(args.input_dim, args.input_dim, 3),
                            gpu=args.gpu)

    # Check if it is pre-computed
    if args.model == 'se_resnet50':
        se_resnet50 = create_model(model=args.model,
                                   input_dim=(args.input_dim, args.input_dim, 3),
                                   pooling=None,
                                   feature_dim=args.feature_dim,
                                   num_classes=args.num_classes)
        if args.opt == 'sgd':
            opt = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
        else:
            opt = Adam()

        se_resnet50.compile(optimizer=opt,
                            loss='categorical_crossentropy',
                            metrics=['acc'])

        stop = EarlyStopping(patience=5)
        filepath = "weights-{epoch:02d}-{val_acc:.2f}.hdf5"
        check_point = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                      save_best_only=True, period=5)

        se_resnet50.fit(x=dataset,
                        batch_size=args.batch_size,
                        epochs=500,
                        callbacks=[stop, check_point],
                        steps_per_epoch=len(file_names)//args.batch_size,
                        validation_data=test_dataset,
                        validation_steps=len(test_file_names)//args.batch_size)

        print('Saving model to disk...')
        se_resnet50.save_weights('model.h5')








