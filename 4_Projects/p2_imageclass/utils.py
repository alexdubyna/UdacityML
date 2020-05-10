import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import json


###############  Readme ###############
#file contains all functions required to prepare data, train/calidate neural network and use it for prediction

######################################    data  utils     ##########################



def prepare_data(data_dir):
    """Takes data directory as input and prepares dataloaders for a model"""
    data_dir = data_dir 
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'train_dataloader' : torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True),
    'test_dataloader' : torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True),
    'valid_dataloader' : torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)}
    
    return dataloaders


def load_json(file):
    """ Takes json file and return a dictionary of values"""
    with open(file, 'r') as f:
            cat_to_name = json.load(f)
    return cat_to_name

        
######################################    prediction  utils     ##########################
        
        
def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array"""

    #from PIL import Image
    
    img = Image.open(image)
    #img=image
    
    #resize, where either width or length is max 256 
    size = (256, 256)
    img.thumbnail(size)
    
    #centrecrop to 224x224
    req_size = 224
    w, h = img.size
    img = img.crop(((w-req_size)//2, (h-req_size)//2, (w+req_size)//2, (h+req_size)//2))
    
    #normalize
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
        
    #transpose colour channel to 1st dimension
    img = img.transpose(2,0,1)
    
    img = torch.from_numpy(img).float()
    
    return img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))


    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax 


#get_labels function
def convert_indexes_to_labels(classes_indexes, model, **kwargs):
    
    """convert predicted indexes of a class to label names. Requires model to have indexes to category dictionary & category to name dictionary """
    
    names_dict = kwargs.get('category_names', model.category_names)
    ks_dict = model.class_to_idx
    categ_names = {}

    a = names_dict.keys()
    b = ks_dict.keys()

    for key in b:
        for value in a:
            if key == value:
                categ_names[ks_dict[key]] = names_dict[value]
    #output dictionary has training_data index as key & name as value
    
    #get predicted flower labels instead of predicted indexes
    classes_labels = []
    for i in classes_indexes[0].tolist():
        classes_labels.append(categ_names[i])
        
    return classes_labels

def show_inference_result(test_img, model, **kwargs):
    
    #checking non-mandatory arguments
    category_names = load_json(kwargs.get('category_names'))
    topk = kwargs.get('top_k', 5)
    device = kwargs.get('gpu', 'cpu')
        
    model.to(device)
    # TODO: Display an image along with the top 5 classes
    img = mpimg.imread(test_img) #read image for first plot
    probs, classes = predict(test_img, model, topk) #run model predictions - prepare data for second chart
    if category_names:
        classes_labels = convert_indexes_to_labels(classes, model, category_names = category_names) #prepare labels
    else:
        classes_labels = convert_indexes_to_labels(classes, model) #prepare labels

    #combine chart & image using matlab-style pyplot interface
    fig = plt.figure(figsize=(4,8))

    #create first figure
    plt.subplot(2,1,1) #(rows, columns, panel No)
    plt.axis('off') #no axis for picture
    plt.title(classes_labels[0]) #name is top-1 predicted class
    imgplot = plt.imshow(img)

    #create second figure
    plt.subplot(2,1,2) 
    plt.barh(classes_labels, probs[0].tolist(), align='center', alpha=0.9)
    plt.show()

    model.to('cpu')

######################################      model related functions     ##########################

def load_checkpoint(filepath):
    
    """Reads the model from checkpoint"""
    #load checkpoint dictionary
    checkpoint = torch.load(filepath)
    
    #initiate model & optimizer
    model = models.vgg16(pretrained=True)
    optimizer = optim.Adam(model.classifier.parameters())

    #update initiated model with trained model state for ready-to-be-used state   
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.category_names = checkpoint['category_names']

    
    return model


def predict(image_path, model, topk):
    """ Predict the class (or classes) of an image using a trained deep learning model"""
    
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path)
    #img = torch.from_numpy(img).type(torch.FloatTensor) 
    model.eval()
    #log_probabilities = model.forward(img.view(1,3,224,224))
    log_probabilities = model.forward(img.unsqueeze(0)) #this is a bit more convenient way to add '1' as dimension at index '0'
    probabilities = torch.exp(log_probabilities)
    probs, classes = probabilities.topk(topk, dim=1)
    probs, classes = probs.detach().numpy(), classes.detach().numpy()
    return probs, classes
    

