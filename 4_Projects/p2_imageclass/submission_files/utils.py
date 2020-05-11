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
    
    return dataloaders, train_dataset.class_to_idx


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

######################################      model usage related functions     ##########################

def load_checkpoint(filepath):
    
    """Reads the model from checkpoint"""
    #load checkpoint dictionary
    checkpoint = torch.load(filepath)
    
    #initiate model & optimizer
    if checkpoint['arch'] == 'Vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = False
        #optimizer = optim.Adam(model.classifier.parameters())
        model.classifier = checkpoint['classifier']
    
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = False
        #optimizer = optim.Adam(model.classifier.parameters())
        model.classifier = checkpoint['classifier']
    
    elif checkpoint['arch'] == 'resnet18':
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = False   
        #optimizer = optim.Adam(model.fc.parameters())
        model.fc = checkpoint['classifier']

    #update initiated model with trained model state for ready-to-be-used state   
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    #default category_names, just because i'm lazy to get therm through train-save-load loop without real need
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    model.category_names = cat_to_name
    
    model.eval()
    
    return model #, optimizer


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


#######################    model creation and training functions #######################
#arch, learning_rate, hidden_units, gpu

def create_model(**kwargs):
    
    #getting model parameters
    arch = kwargs.get('arch')
    hidden_units = kwargs.get('hidden_units', 512)
    user_device = kwargs.get('gpu', 'cpu')
    learning_rate = kwargs.get('learning_rate', 0.003)

       
    #get yourself a pretrained model from pytorh set
    if arch == 'Vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
         
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False    

    else:
        print ('You have selected unsupported model architechture')

    #freeze feature weights (important this comes before assigning classifier)

    # Use GPU if it's available when requested by user and prevent from using if not
    if user_device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    #define classifier supported by selected architechture
    if arch == 'Vgg16':
        model.classifier = nn.Sequential(nn.Linear(25088,4096),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(4096,hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(hidden_units, 102), 
                                     nn.LogSoftmax(dim=1))
        
    elif arch == 'resnet18':
        model.fc = nn.Sequential(nn.Linear(512,512),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(512,hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(hidden_units, 102), 
                                     nn.LogSoftmax(dim=1))
    
    elif arch == 'alexnet':
        model.classifier = nn.Sequential(nn.Linear(9216,4096),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(4096,hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(hidden_units, 102), 
                                     nn.LogSoftmax(dim=1))        
    
    else:
        print ('You have selected unsupported model architechture')

    if arch == 'Vgg16': 
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif arch == 'resnet18':
        for param in model.fc.parameters():
            param.requires_grad = True
    elif arch == 'alexnet': 
        for param in model.classifier.parameters():
            param.requires_grad = True
            
    #define loss function
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    if arch == 'Vgg16':
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif arch == 'resnet18':
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    elif arch == 'alexnet':
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    #move model to available device
    model.to(device);

    
    print ("model_architechture: {}".format(arch),
                  "hidden_units: {} ".format(hidden_units),
           "learning_rate: {} ".format(learning_rate),
                  "training_on: {}".format(user_device)          )
       
    return model, criterion, optimizer
    

    
def train_model(data, model, criterion, optimizer, **kwargs):
 
    
    #training loop
    epochs = kwargs.get('epochs', 1)
    user_device = kwargs.get('gpu', 'cpu')
    dataloaders = data
    step=0
    print_every=50 
    
    
    # Use GPU if it's available when requested by user and prevent from using if not
    if user_device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    for e in range(epochs):
        train_loss = 0
        for images, labels in dataloaders['train_dataloader']:

            step += 1
            images, labels = images.to(device), labels.to(device) # Move images and label tensors to the default device
            optimizer.zero_grad() # clear gradients

            log_probabilities = model.forward(images)  #get logarithm of probabilities of a class
            loss = criterion(log_probabilities, labels) #calculate loss
            loss.backward() #backpropagate loss to update weights for next step
            optimizer.step()
            train_loss += loss.item()


            if step % print_every == 0:
                test_loss = 0
                accuracy = 0
                accuracyn = 0
                model.eval() #enable all layers

                with torch.no_grad():

                    for images, labels in dataloaders['valid_dataloader']:

                        #images = images.view(images.shape[0], 3, -1) #flatten image into vector
                        #feed-forward validation data through current state of model
                        images, labels = images.to(device), labels.to(device) #move data to available device
                        log_probabilities = model.forward(images) #get log probs
                        loss = criterion(log_probabilities, labels) #get test loss
                        test_loss += loss.item()

                        #calculate accuracy for current state of model
                        probabilities = torch.exp(log_probabilities) #predicted class probabilities
                        top_p, top_class = probabilities.topk(1, dim=1) #get top-1 predicted class & its' probabilities
                        equals = top_class == labels.view(*top_class.shape) #compare predicted to real labels
                        accuracy += torch.mean(equals.type(torch.FloatTensor)) #calculate accuracy
                        #revisit this bit on video, not sure accuracy formula stays same for top-5 classes accuracy

                        #calculating top-5 classes accuracy
                        #top_p_n, top_class_n = probabilities.topk(3, dim=1)
                        #initiate all-False comparison tensor
                        #equals_n = torch.zeros(top_class_n.shape[0], dtype=torch.bool) 
                        #
                        #for j in range(top_class_n.shape[0]): #for every test row in predictions
                        #    for i in range(top_class_n.shape[1]): #compare every element from topk to see if at least
                        #        if top_class_n[j][i] == labels[j]: #one of those match to the test data picture label
                        #            equals_n[j] = True #and it it matches - update comparison sheet
                        #accuracyn += torch.mean(equals_n.type(torch.FloatTensor)) #calculate accuracy for top-n elements

                model.train() #enable dropout again

                #some output to see state of calculations during training & validation
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss/len(dataloaders['train_dataloader'])),
                  "Validation Loss: {:.3f}.. ".format(test_loss/len(dataloaders['valid_dataloader'])),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid_dataloader']))
                      #,    "Top-n Acc.: {:.3f}".format(accuracyn/len(valid_dataloader))
                     )
    print (  "epochs: {} ".format(epochs),
                  "training_on: {}".format(user_device)          )
    
    print ('model was successfully trained')
    
    
    
def save_model(model, optimizer, train_dataset_ind_labels,  **kwargs):
       
    epochs = kwargs.get('epochs', None)
    arch = kwargs.get('arch', 'Vgg16')
    save_dir = kwargs.get('save_dir', '')
    save_file_name = save_dir + '/' + arch + '_checkpoint.pth'
    
    if arch == 'Vgg16': 
        for param in model.classifier.parameters():
            param.requires_grad = False
    elif arch == 'alexnet': 
        for param in model.classifier.parameters():
            param.requires_grad = False        
    elif arch == 'resnet18':
        for param in model.fc.parameters():
            param.requires_grad = False
    
    model.class_to_idx = train_dataset_ind_labels
    model.to('cpu')
    
    if arch == 'Vgg16':
        clsf = model.classifier
    elif arch == 'resnet18':
        clsf = model.fc
    elif arch == 'alexnet':
        clsf = model.classifier
    
    checkpoint = {'classifier': clsf,
                  'arch': arch,
                  #'category_names': cat_to_name,
                  'epochs': epochs,
                  'class_to_idx': train_dataset_ind_labels,
                  'optimizer_dict': optimizer.state_dict(),
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, save_file_name)
    
    print ("saved_as: {}".format(save_file_name),
                  "epochs: {} ".format(epochs),
          "architechture: {} ".format(arch))
    print ('model was successfully saved')