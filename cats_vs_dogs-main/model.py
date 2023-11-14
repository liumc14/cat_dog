import torch
import torch.nn as nn

from torchvision import models

class img_class(nn.Module):
    def __init__(self):
        super(img_class,self).__init__()
        self.model=models.resnet50(pretrained=False)
        self.model.load_state_dict(torch.load(r"G:\python\cnn_catanddog\resnet50-0676ba61.pth"))
        self.infeature=self.model.fc.in_features
        self.fc = nn.Linear(self.infeature, 2)
    def forward(self,img):
        self.model.fc=self.fc
        result=self.model(img)
        return result