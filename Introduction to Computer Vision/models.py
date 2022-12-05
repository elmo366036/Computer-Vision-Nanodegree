## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # the model is based on the one from the Facial Key Points using Deep CNN but with a smaller FC
        
        self.conv1 = nn.Conv2d(1,  32,  4)     # (224-4+0)/1 + 1 = 225; pool -> 112
        self.conv2 = nn.Conv2d(32, 64,  3)     # (112-3+0)/1 + 1 = 110; pool -> 55
        self.conv3 = nn.Conv2d(64, 128, 2)     # (55-2+0)/1 + 1  = 54;  pool -> 27
        self.conv4 = nn.Conv2d(128,256, 1)     # (27-1+0)/1 + 1  = 27;  pool -> 13
        
        self.pool  = nn.MaxPool2d(2,2)
        
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.3)
        self.drop4 = nn.Dropout(0.4)
        self.drop5 = nn.Dropout(0.5)
        self.drop6 = nn.Dropout(0.6)
        
        self.fc1   = nn.Linear(256*13*13, 1000)
        self.fc2   = nn.Linear(1000,1000)
        self.out   = nn.Linear(1000,136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        x = self.pool(F.relu(self.conv1(x)))   # output (32, 112, 112)
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))   # output (64, 55, 55) 
        x = self.drop2(x)
        x = self.pool(F.relu(self.conv3(x)))   # output (128, 27, 27)
        x = self.drop3(x)
        x = self.pool(F.relu(self.conv4(x)))   # output (256, 13, 13)
        x = self.drop4(x)
        
        x = x.view(x.size(0), -1)              # output (1, 43264)
        x = F.relu(self.fc1(x))                # output (1, 1000)
        x = self.drop5(x)
        x = F.relu(self.fc2(x))                # output (1, 1000)
        x = self.drop6(x)
        x = self.out(x)                        # output (1, 136)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
