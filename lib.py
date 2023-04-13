from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from dataset import AnimalsDataset
from modelCNN import ModelCNN
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor    
from torch.optim import SGD
import argparse
import os
import shutil
from tqdm.autonotebook import tqdm
import numpy as np