import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from model import img_class
from torch.utils.data import DataLoader
from mydataset import mydataset
from torchvision import transforms
from tqdm import tqdm

log_dir=r"G:\python\cnn_catanddog\cats_vs_dogs-main\log"
batch_size=2
num_worker=2
EPOCH=5
LEARNING_RATE, STEP_SIZE, GAMMA = 0.001, 100, 0.1
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),  # convert to (C,H,W) and [0,1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # mean=0; std=1
    ])
    train_dataset=mydataset(transform)
    train_dataloader=DataLoader(train_dataset,batch_size,num_workers=num_worker)



    model=img_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    criterion = nn.CrossEntropyLoss()
    print("start trainging")
    for epoch in tqdm(range(EPOCH)):
        model.train()
        for imgs,labels in train_dataloader:
            print(imgs.shape)
            print(labels)
            imgs=imgs.to(device=device)

            labels=labels.to(device=device)

            optimizer.zero_grad()
            result=model(imgs)
            loss = criterion(result, labels)

            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(),os.path.join(r"G:\python\cnn_catanddog\cats_vs_dogs-main\log",f'model-epoch-{epoch+1}.pth'))
        scheduler.step()

if __name__=='__main__':
    print(device)
    train()