################################################################
## Import Libaries
################################################################

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import time
import os
import cv2
import numpy as np
import string
import utils
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from PIL import Image #For image processing
import csv
import random
import matplotlib.pyplot as plt

torch.manual_seed(1)
device = torch.device('cuda')


################################################################
## Define Learning Parameters
print()
print("Defined Parameters:")
# Prozentual share of whole Dataset for Training
Share_TrainingSet = 0.8     
print("Share of training set compared to whole dataset: %f" %Share_TrainingSet)
LearningRate = 0.01     
print("Learning rate: %f" %LearningRate)
num_train_epochs = 15
print("Number of epochs: %d" %num_train_epochs)

# Defining some helper variables
target = []
number_of_images = 0  
loss_history = []

## Configuration done
print("Configuration done")



#####################################################################
## Create samplers to split dataset into training and validation sets
#####################################################################
# Read out all csv files to determine the amount of images
for signclass in os.listdir('../GTSRB/Final_Training/Images/'):
       # Load .csv file to get file names and class labels
    with open('../GTSRB/Final_Training/Images/'+signclass+'/GT-'+signclass+'.csv') as f:
       reader = csv.reader(f)
       for row in reader:
            if row[0].find("Filename") < 0:
                number_of_images += 1

# Create and shuffle a list with indices for the whole dataset,
# so that the dataloader loads the images in a random order. 
indices = list(range(number_of_images))
random.shuffle(indices)

# Split data_list into lists for test(80%) and validation set(20%)
splitpoint = round(number_of_images * Share_TrainingSet)
train_indices = indices[:splitpoint]
val_indices = indices[splitpoint:]
print("Number of training samples: %d" %len(train_indices))
print("Number of validation samples: %d" %len(val_indices))
train_sampler = data.sampler.SubsetRandomSampler(train_indices)
val_sampler = data.sampler.SubsetRandomSampler(val_indices)

#####################################################################
## Define image transformation
#####################################################################
data_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

#####################################################################
## Load Data for training and validation
#####################################################################
GTSRB_dataset = datasets.ImageFolder(root='/home/stillerf/GTSRB/Final_Training/Images/',
                                           transform=data_transform)
train_data = DataLoader(
    GTSRB_dataset, 
    sampler = train_sampler,
    batch_size = 64, 
    pin_memory = True,
    )

val_data = DataLoader(
    GTSRB_dataset, 
    sampler = val_sampler,
    batch_size = 64, 
    pin_memory = True
    )

#####################################################################
## Definition of the network architecture
#####################################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(20, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(1024,512)  # Input size is 32x32; 2x pooling makes it 8x8. With 16 layer, all in all 1280 features exist.
        self.fc2 = nn.Linear(512, 43) # The dataet contains 43 different classes
      

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x),kernel_size = 2,stride=2))
        x = F.relu(F.max_pool2d(self.conv3(x),kernel_size = 2,stride=2))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 1024)   # Input size is 32x32; 2x pooling makes it 8x8. With 16 layer, all in all 1280 features exist.
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

######################################################################################
## We create the network, shift it on the GPU and define a optimizer on its parameters
######################################################################################
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr = LearningRate, momentum=0.5)


##################################
## Function for training the model
##################################
def train(epoch):
    model.train()
    epoch_sum_loss = 0
    epoch_loss = 0
    nr_of_batches = 0
    for batch_idx, (data, target) in enumerate(train_data):
        # Move the input and target data on the GPU
        data, target = data.to(device), target.to(device)
        # Zero out gradients from previous step
        optimizer.zero_grad()
        # Forward pass of the neural net
        output = model(data)
        # Calculation of the loss function
        loss = F.cross_entropy(output,target)
        # Backward pass (gradient computation)
        loss.backward()
        # Adjusting the parameters according to the loss function
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_indices),
                10. * batch_idx / len(train_indices), loss.item()))
        # Sum up the loss of each batch
        epoch_sum_loss += loss.item()
        nr_of_batches += 1
    # Calculate average loss for the batch
    epoch_loss = epoch_sum_loss / nr_of_batches
    loss_history.append(epoch_loss)
    print("Average loss of done epoch is: %f" %epoch_loss)

    # Plot the learning curve after each period and save it into learning_curve.png
    plt.figure(1) 
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Loss during training epochs')
    plt.grid(True)
    plt.savefig("learning_curve.png")


###############################################################################################
## Function to test the model on training and validation set 
## (Testset is not loaded with Dataset.Imageloader, so it has its own evaluation/test function)
###############################################################################################
def test(set_to_test, length):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in set_to_test:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()          
    test_loss /= length
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, length,
        100. * correct / length))


##########################################################
## Train the model for "x" epochs 
##########################################################
for epoch in range(1, num_train_epochs + 1):
    train(epoch)


##########################################################
## Evaluation the model on training and validation set 
##########################################################
print("Evaluation on training set")
test(train_data, len(train_indices))
print("Evaluation on validation set")
test(val_data, len(val_indices))

##########################################################
## Evaluation on final test set 
## (Load the dataset and test it on the fly)
##########################################################
# Load .csv file to get file names and class labels. Then process each image and predict the class
attributes = []
testdata= []

with open('../GTSRB/Final_Test/Images/GT-final_test.csv') as f:
    reader = csv.reader(f)
    correct = 0
    length = 0
    stopcount = 0
    for row in reader:
        # stopcount += 1
        # if stopcount == 10:
        #     break
        if row[0].find("Filename") < 0:
            attributes = row[0].split(';')
            filename = attributes[0]
            classlabel = int(attributes[7])
            img = Image.open('../GTSRB/Final_Test/Images/' + filename)
            img_tensor = data_transform(img)
            temp_list = []
            temp_list.append(img_tensor)
            data = torch.stack(temp_list)
            data = data.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1] 
            pred_int = int(pred)
            if pred == classlabel:
                correct += 1
            length += 1
    print("Evaluation on test set:")
    print("Samples: %d" %length)
    print("Correct predicted: %d" %correct)
    test_accuracy = correct / length *100
    print('Accuracy: %f' %test_accuracy)


print("Program finished")
