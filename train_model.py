#B3: Trainning
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
from torch.utils.tensorboard import SummaryWriter
from utils import *


def get_args():
    parser = argparse.ArgumentParser(description = "Train a CNN")
    parser.add_argument("--root", "-r", type= str, default= "./project")
    parser.add_argument("--lr", "-l", type= float, default= 1e-3)
    parser.add_argument("--batch-size", "-b", type= int, default = 16)
    parser.add_argument("--num-epochs", "-e", type = int, default = 100)
    parser.add_argument("--log_path", "-lp", type = str, default = "./project/tensorboard")
    parser.add_argument("--save_path", "-sp", type = str, default ="./project/trained_models")
    args = parser.parse_args()
    return args

args = get_args()
batch_size = args.batch_size
num_epochs = args.num_epochs
if __name__ =="__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])
    # root = "./project"
    root = args.root
    train_dataset = AnimalsDataset(root= root, train= True, transform= transform)
    test_dataset = AnimalsDataset(root= root, train= False, transform= transform)
    
    train_dataloader = DataLoader(
        dataset= train_dataset,
        batch_size= batch_size,
        num_workers= 8,
        shuffle= True,
        drop_last= True
    )
    test_dataloader = DataLoader(
        dataset= test_dataset,
        batch_size= batch_size,
        num_workers= 8,
        shuffle= False,
        drop_last= False
    )

    model = ModelCNN(num_classes= 10).to(device)
    criterion= nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr= args.lr, momentum= 0.9)
    num_inter = len(train_dataloader)

    # Tao cac folder de luu tru model trong qua trinh training
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    
    os.mkdir(args.log_path)

    if os.path.isdir(args.save_path):
        shutil.rmtree(args.save_path)
    
    os.mkdir(args.save_path)
    
    writer = SummaryWriter()
    best_loss = 100000

    #Starting trainning
    for epoch in range(num_epochs):
        model.train
        train_loss = []
        progress_bar = tqdm(train_dataloader, colour= "green")
        for idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            #Xac dinh chieu forward:
            outputs = model(images)
            loss_values = criterion(outputs, labels)
            train_loss.append(loss_values.item())   ##Chuyen tensor thanh so .item()

            #Xac dinh backward:
            optimizer.zero_grad()
            loss_values.backward()
            #Optimizer
            optimizer.step()
            # if idx % 20 == 0:
            #     print("Epoch {}: Interation {}/ {} Loss: {}.".format(epoch, idx, num_inter, loss_values))
            progress_bar.set_description("Epoch {}: Interation {}/ {} Loss: {}".format(epoch+1, idx+1, num_inter, np.mean(train_loss)))
            writer.add_scalar("Train/Loss", np.mean(train_loss), idx + epoch * len(train_dataloader))
    
    #Validation==> danh gia mo hinh
    model.eval()

    all_predictions = []
    all_labels = []
    valid_loss = []
    with torch.no_grad():   #CHi lam chieu forword
        for idx, (images, labels) in enumerate(test_dataloader): 
            images = images.to(device)
            labels = labels.to(device)

            #forward pass
            outputs = model(images)
            loss_values = criterion(outputs, labels)
            valid_loss.append(loss_values.item())

            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions)
            all_labels.extend(labels)
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        acc =  accuracy_score(all_labels, all_predictions)

        writer.add_scalar("Validation/Loss", np.mean(valid_loss), epoch)
        writer.add_scalar("Validation/Acc", acc, epoch)
        plot_confusion_matrix(writer, conf_matrix, [i for i in range(10)], epoch)

        print("epoch {}. Accuracy {}".format(epoch, acc))
        # print(conf_matrix)

        #Save model 
    check_point ={
        "epoch": epoch +1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(check_point, os.path.join(args.save_path, "last.pt"))
    #Luu tru epoch loss nho nhat
    if np.mean(valid_loss) < best_loss:
            best_loss = np.mean(valid_loss)
            torch.save(check_point, os.path.join(args.save_path, "best.pt"))