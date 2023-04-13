#Xay dung model
import torch 
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import cv2
import numpy as np

class ModelCNN(nn.Module):
    def __init__(self, num_classes =10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels= 16,kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels= 32, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride =2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 32  , out_channels= 64, kernel_size= 3, stride= 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 64 , out_channels = 128, kernel_size = 3, stride =1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.linear_1 = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.ReLU()
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU()
        )
        self.linear_3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x

if __name__ == '__main__':
    image = torch.rand(8, 3, 32, 32)
    print(image)
    model = ModelCNN()
    predic = model(image)
    print(predic.shape)
