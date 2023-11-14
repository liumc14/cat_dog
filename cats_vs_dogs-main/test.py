import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

writer=SummaryWriter("cats_vs_dogs-main/log")
img=Image.open(r"G:\python\cnn_catanddog\data\test\1.jpg")
print(img)
img_totensor=transforms.ToTensor()
img_tensor=img_totensor(img)
print(img_tensor.shape)
#img_np=np.array(img)
#print(img_np.shape)
writer.add_image("test_picture",img_tensor,0,dataformats='CHW')