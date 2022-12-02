### This repository contains the multiclass classification for CIFAR 10 using the following model architectures in pytorch

- Linear
- AlexNet
- VGG
- ResNet
- ShuffleNet
- Inception
- Xception

### Usage

1. Create conda environment using the conda yml file
conda env create -f environment.yml

2. Navigate to src/data folder

3. Download raw data using ./download_data.sh

4. cd .. and use arguments to control the training process

```
optional arguments:
  -h, --help            show this help message and exit
  --gpu                 Use GPU for training (default: True)
  --train               Train Model (default: False)
  --model_name MODEL_NAME
                        Train Model Type (default: Vanilla_Dense)
  --batch_size [BATCH_SIZE]
                        Batch size for training the model (default: 32)
  --num_workers [NUM_WORKERS]
                        Number of Available CPUs (default: 5)
  --num_epochs [NUM_EPOCHS]
                        Number of Epochs for training the model (default: 10)
  --num_output_classes [NUM_OUTPUT_CLASSES]
                        Number of output classese for CIFAR (default: 10)
  --learning_rate [LEARNING_RATE]
                        Learning Rate for the optimizer (default: 0.01)
  --sgd_momentum [SGD_MOMENTUM]
                        Momentum for the SGD Optimizer (default: 0.5)
  --plot_output_path PLOT_OUTPUT_PATH
                        Output path for Plot (default: ./Plots_)
  --model_path MODEL_PATH
                        Model Path to resume training (default: None)
  --epoch_save_checkpoint [EPOCH_SAVE_CHECKPOINT]
                        Epochs after which to save model checkpoint (default: 5)
  --patience [PATIENCE]
                        Early stopping epoch count (default: 3)
  --pred_model PRED_MODEL
                        Model for prediction; Default is checkpoint_model.pth; 
                        change to ./best_model.pth for 1 sample best model (default: ./checkpoint_model.pth)
  --transforms TRANSFORMS
                        Transforms to be applied to input dataset. options (posterize,sharpness,contrast,equalize,crop,hflip). 
                        comma-separated list of transforms. (default:
                        None)
```

5. Example

```
python main.py  --train  --model_name RESNET
python main.py  --model_name RESNET --RUN_NUMBER 21_52_02_12
```
