# Face-Recognition

Face recognition models trained on VGGFace2 dataset. Download the full files from releases section. ResNet model with 512 dimension is already trained and weights are provided along with other files. To make predictions run the following command:

```
# To see default values, enter python predict.py --help

python predict.py --model=resnet50 \
                  --platform=ijbb \
                  --batch_size=25 \
                  --feature_dim=512 \
                  --cores=8
```

To train a new model on VGGFace2 dataset run the following command:

```
python train.py --model=se_resnet50 \
                --batch_size=1 \
                --feature_dim=1024 \
                --cores=8 \
                --input_dim=224 \
                --opt=adam \
                --gpu=True
```

The dataset assumed to be in outside the project directory. You can also change VGG_PATH in train.py.