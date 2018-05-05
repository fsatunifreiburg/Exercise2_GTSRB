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
print("Learning rate: %f" %Share_TrainingSet)
################################################################
################################################################
## Configuration done
print("Configuration done")
################################################################

# Normalizing an image decreases the difference between the images in brightness and contrast
normalize = transforms.Normalize(mean= [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# # Transform an image to a 32x32 pixel size to work with a consistent format
transform = transforms.Compose([
     transforms.Resize(32),
     transforms.CenterCrop(32),
     transforms.ToTensor(),
     normalize])

train_data = []
target_list = []
target = []
number_of_images = 0   #


# Read out all csv files to determine the amount of images
for signclass in os.listdir('../GTSRB/Final_Training/Images/'):
       # Load .csv file to get file names and class labels
    with open('../GTSRB/Final_Training/Images/'+signclass+'/GT-'+signclass+'.csv') as f:
       reader = csv.reader(f)
       for row in reader:
            if row[0].find("Filename") < 0:
                number_of_images += 1

# Create a and shuffle a list with indices for the whole dataset,
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


# Define image transformation
data_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# Load Data for training and validation
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
    pin_memory = True,
    )

# Definition of the network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)    # FYI: In the lecture I forgot to add the padding, thats why the feature size calculation was wrong
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(1024,512) #64*4*4
        self.fc2 = nn.Linear(512, 43)
      

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv22(x),kernel_size = 2,stride=2))
        x = F.relu(F.max_pool2d(self.conv3(x),kernel_size = 2,stride=2))
        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(self.conv4(x),kernel_size = 2,stride=2))
        x = x.view(-1, 1024)   # Flatten data for fully connected layer. Input size is 32x32, we have 2 pooling layers so we pool the spatial size down to 8x8. With 20 feature maps as the output of the previous conv we have in total 8x8x20 = 1280 features.
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# We create the network, shift it on the GPU and define a optimizer on its parameters
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr = LearningRate, momentum=0.5)

loss_history = []

# This function trains the neural network for one epoch
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
        #loss = nn.CrossEntropyLoss(output,target)
        loss = F.cross_entropy(output,target)
        #loss = F.nll_loss(output, target)
        # Backward pass (gradient computation)
        loss.backward()
        # Adjusting the parameters according to the loss function
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data.dataset),
                100. * batch_idx / len(train_data), loss.item()))
        # Sum up the loss of each batch
        epoch_sum_loss += loss.item()
        nr_of_batches += 1
    # Calculate average loss for the batch
    epoch_loss = epoch_sum_loss / nr_of_batches
    loss_history.append(epoch_loss)
    print("Average loss of done epoch is: %f" %epoch_loss)

    plt.figure(1) 
    plt.plot(np.arange(len(loss_history)), loss_history)

    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Loss during training epochs')
    plt.grid(True)
    plt.savefig("test.png")

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_data:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(val_data.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_data.dataset),
        100. * correct / len(val_data.dataset)))

num_train_epochs = 20
for epoch in range(1, num_train_epochs + 1):
    train(epoch)

# attributes = []
# testdata= []

# # Load .csv file to get file names and class labels
# with open('../GTSRB/Final_Test/Images/GT-final_test.csv') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         if row[0].find("Filename") < 0:
#             attributes = row[0].split(';')
#             filename = attributes[0]
#             classlabel = attributes[7]
#             img = Image.open('../GTSRB/Final_Test/Images/' + filename)
#             img_tensor = transform(img)

#             # Stack each image together with the label into a list
#             testdata.append((img_tensor, classlabel)
#     testset = data.TensorDataset(testdata)

# testset_loader = DataLoader(
#     testset, 
#     shuffle = True,
#     pin_memory = True,
#     )

test()





print("Program finished")















#val_data = []
# val_data = DataLoader(
#     val_data_list, 
#     batch_size=64, 
#     pin_memory=True,
#     transform = transforms.Compose([
#         transforms.Resize(32),
#         transforms.CenterCrop(32),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#     )



                # # Load image corresponding to the filename and transform it into a tensor
                # img = Image.open('../GTSRB/Final_Training/Images/'+signclass+'/' + filename)
                # img_tensor = transform(img)

                # # Stack each image into the whole train_img_list resp. into a "train_data" tensor and the classlabel into the target list concurrently
                # train_img_list.append(img_tensor)
                # target_list.append(classlabel)

                # # To break up after 1000 elements, use following 2 lines
                # #if len(train_data_list) >= 1000:
                # #    break

                # if len(train_data_list) >= 64: 
                #     train_data.append(torch.stack(train_img_list),target_list)
                #     train_data_list = []
                #     target_list = []
                #     print("One Batch full")

    

#     for f in os.listdir('../GTSRB/Final_Training/Images/'+signclass+'/'):
#         # Check if the the file f is a image (ends with .ppm)
#         if f.find('.ppm') >= 0:
#             # Transform each Training Image into an tensor
#             img = Image.open('../GTSRB/Final_Training/Images/'+signclass+'/'+f)
#             img_tensor = transform(img)
#             # Stack each image into the whole train_data_list resp. into a "train_data" tensor
#             train_data_list.append(img_tensor)
#             #if len(train_data_list) >= 1000:
#             #    break
#             if len(train_data_list) >= 64: 
#                 train_data.append(torch.stack(train_data_list))
#                 train_data_list = []
                
            
# train_data = torch.stack(train_data_list)
# print(len(train_data_list))

