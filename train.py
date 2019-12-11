import copy
import numpy as np
import json

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from collections import OrderedDict
import torchvision.models as models

import argparse

parser = argparse.ArgumentParser(description='Train Image Classifier')

parser.add_argument('--arch', type = str, default = 'vgg13', help = 'NN Model Architecture')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning Rate')
parser.add_argument('--hidden_units', type = int, default = 512, help = 'Neurons in the Hidden Layer')
parser.add_argument('--epochs', type = int, default = 20, help = 'Epochs')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'GPU or CPU')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')

arguments = parser.parse_args()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
}
# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}
# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True)
}
class_names = image_datasets['train'].classes


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    

if arguments.arch == 'vgg13':
    input_size = 25088
    model = models.vgg13(pretrained=True)
elif arguments.arch == 'alexnet':
    input_size = 9216
    model = models.alexnet(pretrained=True)

#print(model)
    
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, arguments.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(arguments.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

for param in model.parameters():
    param.requires_grad = False
    
model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = arguments.learning_rate)

if torch.cuda.is_available():
    arguments.gpu = 'cuda'
else:
    arguments.gpu = 'cpu' 

def validation(model, validateloader, criterion):
    
    val_loss = 0
    accuracy = 0
    
    for images, labels in iter(validateloader):

        images, labels = images.to(arguments.gpu), labels.to(arguments.gpu)

        output = model.forward(images)
        val_loss += criterion(output, labels).item()

        probabilities = torch.exp(output)
        
        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return val_loss, accuracy

epochs = arguments.epochs
steps = 0

for e in range(epochs):
    model.to(arguments.gpu)
    model.train()
    running_loss = 0
    running_corrects = 0
    for images, labels in dataloaders['train']:
        steps += 1
        images = images.to(arguments.gpu)
        labels = labels.to(arguments.gpu)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
        
        if steps % 40 == 0:
            model.eval()
            
            with torch.no_grad():
                validation_loss, accuracy = validation(model, dataloaders['valid'], criterion)

            print('Epoch : {}/{}'.format(e+1, epochs))
            print('-' * 10)
            print("\n Training loss :",running_loss/40)#len(dataloaders['train']))
            print("\n Validation Loss:", validation_loss/len(dataloaders['valid']))
            print("\n Validation Accuracy :", accuracy/len(dataloaders['valid']))
            print("\n")
            running_loss = 0
            model.train()
def test_accuracy(model, test_loader):

    # Do validation on the test set
    model.eval()
    model.to(arguments.gpu)

    with torch.no_grad():
    
        accuracy = 0
    
        for images, labels in iter(test_loader):
    
            images, labels = images.to(arguments.gpu), labels.to(arguments.gpu)
    
            output = model.forward(images)

            probabilities = torch.exp(output)
        
            equality = (labels.data == probabilities.max(dim=1)[1])
        
            accuracy += equality.type(torch.FloatTensor).mean()
        
        print("Test Accuracy: {}".format(accuracy/len(test_loader)))    
        
        
test_accuracy(model, dataloaders['test'])
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'arch': arguments.arch,
                  'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict()
                 }
torch.save(checkpoint, arguments.save_dir)

