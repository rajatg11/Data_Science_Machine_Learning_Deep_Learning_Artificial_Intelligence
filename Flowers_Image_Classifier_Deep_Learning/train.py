# Imports here
import torch
from torchvision import datasets, models, transforms
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
import argparse
import json

#To get the input from command line 
parser = argparse.ArgumentParser(description='Training the model')

#Following are the inputs can be received from command line
parser.add_argument('data_directory', type=str, help='Dataset path')
parser.add_argument('--checkpoint', type=str, help='checkpoint name')
parser.add_argument('--arch', type=str, default='vgg16' ,help='Enter the model architecture name vgg16 or Alexnet or ResNet101')
parser.add_argument('--hidden_units', type=int, default=1024 ,help = 'Enter the hidden unit')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Enter the learning rate')
parser.add_argument('--epochs', type=int, default=4, help='Enter Epochs')
parser.add_argument('--gpu',type=str,default='y', help='Enter Y or N')

#This line parses the arguments passed from command line
args = parser.parse_args()

#Get the Data from data_directory passed from command line
data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([.485,.456,.406],
                                                           [.229,.224,.225])
                                      ])
test_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([.485,.456,.406],
                                                           [.229,.224,.225])
                                      ])
validation_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([.485,.456,.406],
                                                           [.229,.224,.225])
                                      ])
# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
validate_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
validloader = torch.utils.data.DataLoader(validate_data, batch_size=32)

#check for epochs entered, by default 4 is defined
if args.epochs:
    epochs = args.epochs
print('epochs are',epochs)

#Check for arch entered, be default 'VGG16' will be loaded
if args.arch:
    if args.arch == 'vgg16' or args.arch == 'alexnet':
        arch = args.arch
    else:
        raise ValueError('Unexpected Architecture entered')

#check for learning rate entered, by default 0.01 will be considered
if args.learning_rate:
    learning_rate = args.learning_rate
    
#check for hidden_units entered, by default 1024 will be considered
if args.hidden_units:
    hidden_units = args.hidden_units
 
#check for gpu or cpu, by default gpu will be considered as yes
if args.gpu:
    if args.gpu == 'Y' or args.gpu =='y':
        device = 'gpu'
    elif args.gpu =='N' or args.gpu =='n':
        device = 'cpu'
    else:
        raise ValueError('Invalid value entered for device')

#import cat to name json file
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#load_model function loads the model and define the classifier
def load_model(arch, hidden_units):
    # Load the model
    if arch == 'vgg16':
        print('vgg16 model is being loaded')
        model = models.vgg16(pretrained = True)
    elif arch == 'alexnet':
        print('alexnet model is being loaded')
        model = models.alexnet(pretrained = True)
    else:
        raise ValueError('Unexpected network architecture', arch)

    #freeze the paramaters
    for param in model.parameters():
        param.requires_grad = False
 
    print('hidden units are', hidden_units)
    
    #define the classifier
    features = list(model.classifier.children())[:-1]
    num_filters = model.classifier[len(features)].in_features
    num_labels = len(train_data.classes)
    if arch == 'vgg16':
        features.extend([nn.Linear(num_filters, hidden_units),
                         nn.ReLU(),
                         nn.Dropout(p=0.5),
                         nn.Linear(hidden_units, num_labels)])
    elif arch == 'alexnet':
        features.extend([nn.Dropout(p=0.5),
                         nn.Linear(num_filters, hidden_units),
                         nn.ReLU(),
                         nn.Linear(hidden_units, num_labels)])
    
    features.extend([nn.LogSoftmax(dim=1)])
                        
    model.classifier = nn.Sequential(*features)
      
    return model

def check_accuracy_on_validation(model, validloader, criterion, device):
    
    validation_loss = 0
    accuracy = 0
   
    model.eval()
    
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            if device =='gpu':
                images, labels = images.to('cuda'), labels.to('cuda')
            
            output = model.forward(images)
            validation_loss += criterion(output, labels).item()
            _, predicted = torch.max(output.data, 1)
            #total = labels.size(0)
            #correct = (predicted == labels).sum().item()
            #accuracy = correct/total
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    return validation_loss, accuracy 

#deep_learning_train function actually trains the model
def deep_learning_train(model, trainloader, validloader, epochs, print_every, device, learning_rate):
    #define the loss
    criterion = nn.NLLLoss()
    
    #learning rate
    print('learning rate is',learning_rate)

    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, model.classifier.parameters())), lr=learning_rate)
    
    if device == 'gpu': 
        if torch.cuda.is_available():
            print('model is being trained on', device) 
        else:
            raise ValueError('Cuda is not available on the device')
    else:
        print('model is being trained on', device) 

    epochs = epochs
    print_every = print_every
    steps = 0
    
    #device_to_use = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device =='gpu':
        model.to('cuda')
    else:
        model.to('cpu')
    
    model.train()
    for e in range(epochs):
        
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            if device == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()

            #Forward and backward passes

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, accuracy = check_accuracy_on_validation(model, validloader, criterion, device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0
                model.train()
                
    return model, optimizer
    
# TODO: Do validation on the test set

    #print('Accuracy of the network on the 819 test images: %d %%' % (100 * correct / total))

# execute the functions
print_every = 40

#load the model
model =  load_model(arch, hidden_units)
#train the model
model, optimizer = deep_learning_train(model, trainloader, validloader, epochs, print_every, device, learning_rate)


# TODO: Save the checkpoint 
model.class_to_idx = train_data.class_to_idx
#convert the model to CPU
if device == 'gpu':
    model.cpu()

#save the checkpoint    
checkpoint = {'arch':arch,
              'epochs':epochs,
              'state_dict': model.state_dict(),
              'optimizer':optimizer.state_dict(),
              'class_to_idx':model.class_to_idx,
              'hidden_units':hidden_units
             }
if args.checkpoint:
    checkpoint_name = args.checkpoint + '.pth'
else:
    checkpoint_name = 'checkpoint.pth'
    
torch.save(checkpoint, checkpoint_name)