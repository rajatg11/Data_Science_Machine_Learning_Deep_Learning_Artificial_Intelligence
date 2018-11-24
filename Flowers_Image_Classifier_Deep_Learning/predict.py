# Imports here
import torch
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
import seaborn as sns
import argparse
import json

parser = argparse.ArgumentParser(description='predict the flower')
parser.add_argument('input', type=str, help='Input path to Image')
parser.add_argument('checkpoint', type=str, help='Checkpoint Name')
parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
parser.add_argument('--category_names', type=str, help='Use a mapping of categories to Real names')
parser.add_argument('--gpu', type=str, default='cpu', help='enter y or n for gpu')

args = parser.parse_args()

#Get the input path
path = args.input
print('Path of the image to predict is',args.input)
#Get the checkpoint name
checkpoint = args.checkpoint
print('checkpoint is',checkpoint)
#Check if topk has been entered, otherwise default to 5
if args.top_k:
    topk = args.top_k

print('Number of most likely classes to return', topk)

#Check if mapping to category name has been entered
if args.category_names:
    import json
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
print('following are the flowers categories')
print(cat_to_name)

#Check for gpu
if args.gpu == 'y' or args.gpu == 'Y':
    device = 'gpu'

#Load the checkpoint from file entered from command line
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

def load_model(checkpoint):
	#restore the value of arch and hidden_units
	arch = checkpoint['arch']
	hidden_units = checkpoint['hidden_units']

	#load the model
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

	model.class_to_idx = checkpoint['class_to_idx']
	#define the classifier
	features = list(model.classifier.children())[:-1]
	num_filters = model.classifier[len(features)].in_features
	num_labels = len(model.class_to_idx)
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

	model.load_state_dict(checkpoint['state_dict'])
	#optimizer.load_state_dict(checkpoint['optimizer'])
	epochs = checkpoint['epochs']
	
	return model

model = load_model(checkpoint)

#following is the function to process the image
from PIL import Image
def process_image(image, device):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    img_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224), transforms.ToTensor()])
    im = img_transforms(im).float()
    if torch.cuda.is_available() and device == 'gpu':
        im = im.to('cuda')
    np_image = np.array(im)
    
    means = np.array([.485,.456,.406])
    std = np.array([.229,.224,.225])
    
    np_image = (np.transpose(np_image, (1, 2, 0)) - means)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image

#img = process_image(path)
#print('image', img)

def predict_prob(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path, device)
   
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    img = image_tensor.unsqueeze_(0)
    
    with torch.no_grad():
        # as model has been loaded as model.cpu() so below line has been commented
        #output = model.forward(img.cuda())
        output = model.forward(img)
        
    top_probs, top_labs = torch.exp(output).topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
         
    return top_probs, top_labs, top_labels, idx_to_class
    
probs, labs, labels, idx_to_class = predict_prob(path, model, topk, device)

print('top {0} probablities are'.format(topk))
print(probs)
print('top {0} labels are'.format(topk))
print(labels)

if cat_to_name:
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in labs]
print('top {0} flowers are'.format(topk))
print(top_flowers)