import argparse
import utils
import os

################## Readme #################
#file configures set of commands for command line training of picture classifier using a new neural network 

# collect input arguments
parser = argparse.ArgumentParser (description = "Train your own image recognition neural network")

parser.add_argument ('data_dir', help = 'Data directory for model', type = str)
parser.add_argument ('--save_dir', help = 'Provide saving directory', type = str, default = os.getcwd())
parser.add_argument ('--arch',choices = ['Vgg16', 'resnet18', 'alexnet'], help = 'Vgg16 is default, alternatively use any other allowed options', type = str, default = 'Vgg16')
parser.add_argument ('--learning_rate', choices = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.05], help = 'Learning rate, default is 0.003', type = float, default = 0.003)
parser.add_argument ('--hidden_units', choices = [512, 256, 128], help = 'Hidden units in Classifier, default is 512', type = int, default = 512)
parser.add_argument ('--epochs', choices = [1,2,3,5,8,13,21,34], help = 'Epochs, default is 1', type = int, default = 1)
parser.add_argument ('--gpu',  choices = ['cuda', 'cpu'], help = "enable GPU calc, default is cpu, use 'cuda' for gpu", type = str, default = 'cpu')

#resulting dictionary of arguments
args_train = vars(parser.parse_args()) 

#####################    Training the model      ####################


#prepare data from data_dir
dataloaders, train_dataset_ind_labels = utils.prepare_data(args_train['data_dir'])

#prepare model using user inputs
model, criterion, optimizer = utils.create_model(arch = args_train['arch'],
                          hidden_units = args_train['hidden_units'],
                          learning_rate = args_train['learning_rate'],   
                          gpu = args_train['gpu'])

#train & validate model
utils.train_model(dataloaders, model, criterion, optimizer,
                  epochs = args_train['epochs'],
                  gpu = args_train['gpu'])

#save model to save_dir
utils.save_model(model, optimizer, train_dataset_ind_labels,
                 epochs = args_train['epochs'],
                 save_dir = args_train['save_dir'],
                arch = args_train['arch'])


###################        test line  for terminal run    ##########################

#python3 train.py 'flower_data_small_set'
#python3 train.py 'flower_data_small_set' --arch 'resnet18' --learning_rate 0.05 --hidden_units 128 --epochs 1
#python3 train.py 'flower_data_small_set' --arch 'Vgg16' --learning_rate 0.05 --hidden_units 128 --epochs 1
#python3 train.py 'flower_data_small_set' --arch 'alexnet' --learning_rate 0.002 --hidden_units 256 --epochs 5











