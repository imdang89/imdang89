#B1.Tao Dataset
import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from torchvision.transforms import ToPILImage, Resize, ToTensor, Compose
import torch.nn
import PIL
import numpy as np
from torch.optim import SGD
from sklearn.metrics import confusion_matrix, accuracy_score


class AnimalsDataset(Dataset):
    def __init__(self, root ="", train = True, transform = None):
        path = os.path.join( root, "animals")
        if train:
            data_path = os.path.join(path, "train")
        else:
            data_path = os.path.join(path, "test")
        catgories = os.listdir(data_path)
        self.transform = transform 
        self.images_path = []
        self.labels_path = []
        for idx, item_name in enumerate(catgories):
            sub_path = os.path.join(data_path, item_name)
            if os.path.isdir(sub_path):
                for item in os.listdir(sub_path):
                    if ".txt" in item or ".mp4" in item:
                        continue
                    img_path = os.path.join(sub_path, item)
                    self.images_path.append(img_path)
                    self.labels_path.append(idx)

    def __len__(self):
        return len(self.labels_path)
    def __getitem__(self, idx):
        image = cv2.imread(self.images_path[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image= self.transform(image)
        label = self.labels_path[idx]
        return image, label
    

if __name__ == "__main__":
    path = "project"
    data = AnimalsDataset(root= path, train= True)
    image , label = data.__getitem__(23000)
    print(data.__len__())
    print(image, label)