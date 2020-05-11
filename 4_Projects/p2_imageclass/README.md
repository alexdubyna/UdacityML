# AI Programming with Python Project

This repo contains ready project code for Udacity's AI Programming with Python Nanodegree program. In this project, code was developed for an image classifier built with PyTorch, then converted into a command line application.

Main files:
##Image Classifier Project.ipynb
Jupyter notebook with code to train a neural network

##utils.py
main file with all functions, following the logic used in jupyter notebook
##train_py
wrapper using utils.py to train a network from command line

Train your own image recognition neural network

positional arguments:
  data_dir              Data directory for model

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Provide saving directory
  --arch {Vgg16,resnet18,alexnet}
                        Vgg16 is default, alternatively use any other allowed
                        options
  --learning_rate {0.0005,0.001,0.002,0.003,0.005,0.01,0.05}
                        Learning rate, default is 0.003
  --hidden_units {512,256,128}
                        Hidden units in Classifier, default is 512
  --epochs {1,2,3,5,8,13,21,34}
                        Epochs, default is 1
  --gpu {cuda,cpu}      enable GPU calc, default is cpu, use 'cuda' for gpu

##predict.py
wrapper using utils.py to test network

Test your own image recognition neural network

positional arguments:
  path_to_image         Path to test image for prediction
  path_to_model_checkpoint
                        Path to model checkpoint to use

optional arguments:
  -h, --help            show this help message and exit
  --category_names CATEGORY_NAMES
                        Category names for predicted classes
  --top_k {1,2,3,4,5,6,7,8,9}
                        Top-N predicted classes
  --gpu {cuda,cpu}      to enable gpu calc use 'cuda'
