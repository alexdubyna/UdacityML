import argparse
import utils
import os

################## Readme #################
#file configures set of commands for command line training of picture classifier using a new neural network 

# collect input arguments
parser = argparse.ArgumentParser (description = "Train your own image recognition neural network")

parser.add_argument ('data_dir', help = 'Data directory for model', type = str, required = True)
parser.add_argument ('--save_dir', help = 'Provide saving directory. Optional argument', type = str, default = os.getcwd())
parser.add_argument ('--arch', help = 'Vgg16 is default, alternatively use any from torchvision.models for options', type = str, default = 'Vgg16')
parser.add_argument ('--learning_rate', help = 'Learning rate, default is 0.003', type = float, default = 0.003)
parser.add_argument ('--hidden_units', help = 'Hidden units in Classifier, default is 512', type = int, default = 512)
parser.add_argument ('--epochs', help = 'Epochs, default is 10', type = int, default = 10)
parser.add_argument ('--gpu', help = "enable GPU calc, default is cpu, use 'cuda' for gpu", type = str, default = 'cpu')

#resulting dictionary of arguments
args_train = vars(parser.parse_args()) 


#TODO: finish training part of the program














