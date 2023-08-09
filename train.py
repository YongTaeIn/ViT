## Load Module

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
from torch.utils.data import random_split


# load custom module
from layers.patch_embedding import PatchEmbedding
from layers.Mlp_head import ClassificationHead
from layers.Earlystopping import EarlyStopping
from block.Encoder_Block import TransformerEncoder


# ## 알파(투명도) 채널 때문에 제거하려고 수행함. 
# # 이미지를 열어서 알파 채널을 제거

from PIL import Image  # RGBA (alpha 채널 제거 방법.)
# image = Image.open("./cat.png")
# image = image.convert("RGB")

# # 필요에 따라 이미지 크기를 변경 (ImageNet 크기로 변경하려면)
# image = image.resize((224, 224))

# # 이미지를 NumPy 배열로 변환 (옵셔널)

# image_array = np.array(image)

# # 이미지를 Pillow 이미지 객체로 변환 (옵셔널)
# image = Image.fromarray(image_array)

# # 알파 채널이 제거된 이미지를 저장하거나 처리에 사용
# image.save("./catt.png")



#############  모델
class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 10,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
############# 모델



##############################################################
###################### CIFAR-10 데이터 다운 및 로드 ##############
##############################################################

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224,224), antialias=True),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# 학습 데이터를 일정 비율로 나누어 validation set을 생성합니다
val_size = int(len(trainset) * 0.2)  # 예시로 학습 데이터의 20%를 validation set으로 사용
train_size = len(trainset) - val_size
trainset, valset = random_split(trainset, [train_size, val_size])

# 데이터 로더를 생성합니다
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

## test loader
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
##############################################################
###################### train, valid 평가 함수. ##############
##############################################################


def train(model,train_loader,optimier,log_interval):
    # train 으로 설정. 
    model.train()
    for batch_idx,(image,label) in enumerate(train_loader):
        image=image.to(device)
        label=label.to(device)

        # 그라디언트 0으로 초기화 
        optimizer.zero_grad()
        output=model(image)
        loss=criterion(output,label)
        loss.backward()

        # 파라미터 업데이트 코드.
        optimizer.step()

        if batch_idx %log_interval ==0:
            print("Train Epoch: {}[{}/{}({:.0f}%)]\t Train Loss : {:.6f}".format
                  (epochs,batch_idx*len(image),len(train_loader.dataset),100*batch_idx/len(train_loader),
                   loss.item()))
    

def evaluate(model,test_loader):
    #평가로 설정.
    model.eval()
    test_loss=0
    correct=0
    
    # 자동으로 gradient 트래킹 안함. 
    with torch.no_grad():
        for batch_idx,(image,label) in enumerate(test_loader):
            image=image.to(device)
            label=label.to(device)
            output=model(image)
            test_loss+=criterion(output,label).item()
            prediction=output.max(1,keepdim=True)[1]
            correct+=prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /=len(test_loader.dataset)
    test_accuracy=100.* correct/len(test_loader.dataset)
    return test_loss,test_accuracy


##############################################################
###################### 모델 파라미터 설정. ##############
##############################################################


device = torch.device('cuda:2')

vit=ViT(in_channels = 3,
         patch_size = 16,
        emb_size = 768,
        img_size= 224,
        depth = 6,
        n_classes = 10).to(device)

epochs=1000
lr=0.001
patience=10

early_stopping = EarlyStopping(patience = patience, verbose = True)




criterion = nn.CrossEntropyLoss()  # loss 함수 
optimizer = optim.SGD(vit.parameters(), lr=lr, momentum=0.9) # 최적화 함수.




os.makedirs('./pt', exist_ok=True)

best_val_loss = float('inf')  # Initialize with a large value to ensure the first validation loss will be lower

for epoch in range(1,epochs+1):
    train(vit,trainloader,optimizer,log_interval=5)
    test_loss,test_accuracy=evaluate(vit,valloader)
    print("\n[Epoch: {}],\t Test Loss : {:.4f},\tTest Accuracy :{:.2f} % \n".format
          (epoch, test_loss,test_accuracy))

    # test_loss 로 설정하면 val loss,  
    # -test_accuracy 로 설정하면 val_accuracy로 동작함. 
    early_stopping(test_loss, vit)
        
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        # Save the model when the validation loss improves
        model_path = f"{'./pt/'}model_epoch_{epoch}_Accuracy_{test_accuracy:.2f}.pt"
        torch.save(vit.state_dict(), model_path)    

    if early_stopping.early_stop:
        print("Early stopping")
        break

