import argparse
import utils

################## Readme #################
#file configures set of commands for command line predicitons of picture classes using a trained neural network


####################        collect parameters from command line         #########################

# collect input arguments
parser = argparse.ArgumentParser (description = "Test your own image recognition neural network")

parser.add_argument ('path_to_image', type = str, help = 'Path to test image for prediction')
parser.add_argument ('path_to_model_checkpoint', type = str, help = 'Path to model checkpoint to use')
parser.add_argument ('--category_names', type = str, default = 'cat_to_name.json', help = 'Category names for predicted classes')
parser.add_argument ('--top_k', type = int, choices = range(1,10), default = 5, help = 'Top-N predicted classes')
parser.add_argument ('--gpu', type = str, choices = ['cuda', 'cpu'], default = 'cpu', help = "to enable gpu calc use 'cuda'")

#resulting dictionary of arguments
args_predict = vars(parser.parse_args()) 


#####################     inference result shown to user      ####################
model = utils.load_checkpoint(args_predict['path_to_model_checkpoint']) #load model from save
utils.show_inference_result(args_predict['path_to_image'], model,
                            top_k = args_predict['top_k'],
                           gpu = args_predict['gpu'],
                           category_names = args_predict['category_names'])




###################        test line  for terminal run    ##########################
#don't forget to 'source activate ENV' you are working with

#python3 predict.py 'testpicimage_2.jpg' checkpoint_vgg16_flowers_small_dataset_model.pth --top_k 3 --category_name test_cat.json --gpu 'cuda'