import numpy as np
import json

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from collections import OrderedDict
import torchvision.models as models
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sb

#import os
#os.environ['QT_QPA_PLATFORM']='offscreen'

import argparse

parser = argparse.ArgumentParser(description='Image Classifier Predictions')

# Command line arguments
parser.add_argument('--image_dir', type = str, default = 'flowers/test/15/image_06351.jpg', help = 'Path to image')
parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')
parser.add_argument('--topk', type = int, default = 5, help = 'Top k classes and probabilities')
parser.add_argument('--json', type = str, default = 'cat_to_name.json', help = 'category_to_name json file')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'GPU or CPU')

arguments = parser.parse_args()

with open(arguments.json, 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'alexnet':  
        model = models.alexnet(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    else:
        print("Architecture not recognized.")
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 512)),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(512, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

# Load pretrained network
model = load_checkpoint(arguments.checkpoint)
print(model)

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image_path)
    
    # Resize
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((5000, 256))
    else:
        pil_image.thumbnail((256, 5000))
        
    # Crop 
    left_margin = (pil_image.width-224)/2
    bottom_margin = (pil_image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Normalize
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
    # Color channel needs to be first; retain the order of the other two dimensions.
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=arguments.topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(arguments.gpu)
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    
    # Convert image to PyTorch tensor first
    if (arguments.gpu == 'cuda'):
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        image = torch.from_numpy(image).type(torch.FloatTensor)
    
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)
    
    output = model.forward(image)
    
    probabilities = torch.exp(output)
    
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)
    
    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    #print(idx_to_class)
    
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes

probs, classes = predict(arguments.image_dir, model)   
print(probs)
print(classes)


# Plot flower input image
plt.figure(figsize = (6,10))
plot_1 = plt.subplot(2,1,1)

image = process_image(arguments.image_dir)

key = arguments.image_dir.split('/')[-2]

flower_title = cat_to_name[key]

imshow(image, plot_1, title=flower_title);

# Convert from the class integer encoding to actual flower names
flower_names = [cat_to_name[i] for i in classes]

# Plot the probabilities for the top 5 classes as a bar graph
plt.subplot(2,1,2)

sb.barplot(x=probs, y=flower_names, color=sb.color_palette()[0]);

plt.show()
