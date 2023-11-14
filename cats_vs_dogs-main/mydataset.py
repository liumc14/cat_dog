import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
rootdata_dir=r"G:\python\cnn_catanddog\data"

class mydataset(Dataset):
    def __init__(self,transform=None,train=True):
        self.transform=transform
        self.train=train
        if train:
            self.data_dir=os.path.join(rootdata_dir,"train")
            self.img_list=os.listdir(self.data_dir)
        else:
            self.data_dir=os.path.join(rootdata_dir,"test")
            self.img_list = os.listdir(self.data_dir)
    def __getitem__(self, idx):
        if self.train:
            label=self.img_list[idx].split('.')[0]
            if label=='cat':
                ans=0
            else:
                ans=1
        else:
            label=""
        img=Image.open(os.path.join(self.data_dir,self.img_list[idx])).convert('RGB')
        nparray_img=np.array(img)
        tensor_img=self.transform(img)

        return tensor_img,ans
    def __len__(self):
        return len(self.img_list)
