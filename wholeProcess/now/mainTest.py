import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils


import tqdm

from dataSetLoad import Hyper_IMG
from autoEncoder import Autoencoder
from retrieve import HybridSN
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = Hyper_IMG(r'E:\my_project\hyperspectralData\medician\leaf',train="Train",transform=transforms.Compose([transforms.ToTensor()])) # , transforms.Normalize(0.5, 0.5, 0)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

model = HybridSN(rate=16, class_num=1, windowSize=128, K=204)

modelname = 'main.pth'
try:
    model.load_state_dict(torch.load(modelname))
    print('[INFO] Load Model complete')
except:
    pass

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.00001)


trainAutoEncoder = True
# 训练模型
if trainAutoEncoder:
    num_epochs = 100
    lossMin = 2
    for epoch in range(num_epochs):
        # model.train()
        model.to(device)
        lossTatol = 0
        t = tqdm.tqdm(enumerate(train_loader),desc = f'[train]') # ,loss:{loss.item()}
        for step, (img, label1, label2) in t:
            output = model(img.permute(0, 1, 4, 2, 3).to(device))
            if math.isnan(output[0][0]):
            # if 0 in img:
                print(output)
                print(img.shape)
                print(label1)
                print(step)
                # for i in img[0,0]:
                #     for j in i:
                #         for k in j:
                #             print(k)
                break
            loss = criterion(output, label1.to(device))
            lossTatol += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if math.isnan(output[0][0]):
            # print(output)
            break
            
        lossAverage = lossTatol/(step+1)
        print('Epoch [{}/{}], AverageLoss: {:.4f}, loss: {:.4f}'.format(epoch+1, num_epochs, lossAverage, loss.item()))



