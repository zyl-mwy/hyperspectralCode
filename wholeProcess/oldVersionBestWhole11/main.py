import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataSetLoad import Hyper_IMG
from retrieve import HybridSN
from myDefine import setup_seed, rule
from myLoss import R2Loss
from myTest import test


trainModel = True
num_epochs = 100
modelname = 'main.pth'

setup_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = Hyper_IMG(r'E:\my_project\hyperspectralData\medician\leaf',train="Train",transform=transforms.Compose([transforms.ToTensor()
                                                                                                                      # ,transforms.RandomHorizontalFlip(p=0.5)
                                                                                                                      # ,transforms.RandomVerticalFlip(p=0.5)
                                                                                                                      ])) # , transforms.Normalize(0.5, 0.5, 0)
val_dataset = Hyper_IMG(r'E:\my_project\hyperspectralData\medician\leaf',train="Val",transform=transforms.ToTensor())
test_dataset = Hyper_IMG(r'E:\my_project\hyperspectralData\medician\leaf',train="Test",transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False) # 32
val_loader = DataLoader(dataset=val_dataset, batch_size=6, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

model = HybridSN(rate=16, class_num=2, windowSize=16, K=204)

criterion = nn.MSELoss()
criterion1 = R2Loss()

optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': 0.001}], lr=0.001) # 0.000001 0.001 , weight_decay=1e-6
# scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = rule)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,
                                                            eta_min=0, last_epoch=-1)
        
# 955 168

if trainModel:
    lossMin = 2
    R2Max, R21Max, R22Max = -200, -200, -200
    for epoch in range(num_epochs):
        model.train()
        model.to(device)
        lossTatol = 0
        t = tqdm.tqdm(enumerate(train_loader),desc = f'[train]') # ,loss:{loss.item()}
        for step, (img, label) in t:
            output = model(img.permute(0, 1, 4, 2, 3).to(device))
            
            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))
            loss = criterion1(output, label.to(device)) + criterion(output, label.to(device)) + 1e-7 * regularization_loss
            # loss = criterion1(output, label.to(device)) + criterion(output, label.to(device))
            
            lossTatol += loss.item()
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
            
            
        lossAverage = lossTatol/(step+1)
        print('Epoch [{}/{}], AverageLoss: {:.4f}, loss: {:.4f}, lr: {}'.format(epoch+1, num_epochs, lossAverage, loss.item(), optimizer.state_dict()['param_groups'][0]['lr'])) # optimizer.state_dict()['param_groups'][0]['lr'])


        test_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
        outputList = []
        labelList = []
        model.eval()

        # 模型测试
        lossTatol = 0
        t = tqdm.tqdm(enumerate(test_loader),desc = f'[Test]') # ,loss:{loss.item()}
        for step, (img, label) in t:
            output = model(img.permute(0, 1, 4, 2, 3).to(device))
            # loss = criterion(output, label.to(device))
            loss = criterion(output, label.to(device)) + criterion1(output, label.to(device))
            
            lossTatol += loss.item()
            outputList.append(np.array(output[0].cpu().detach().numpy()))
            labelList.append(np.array(label[0].cpu().detach().numpy()))

        lossAverage = lossTatol/(step+1)
        outputList = np.array(outputList)
        labelList = np.array(labelList)

        R2 = r2_score(outputList, labelList) 
        R21 = r2_score(outputList[:, 0], labelList[:, 0])
        R22 = r2_score(outputList[:, 1], labelList[:, 1])
        if R2 > R2Max:
            # R2Max = R22+R21
            R2Max = R2
            R21Max = R21
            R22Max = R22
            torch.save(model.state_dict(),modelname)
            print('Model Saved!! R2Max: {:.4f}, R21Max: {:.4f}, R22Max: {:.4f}'.format(R2Max, R21Max, R22Max))
        else:
            print('R2Max: {:.4f}, R21Max: {:.4f}, R22Max: {:.4f}'.format(R2Max, R21Max, R22Max))
        print(' ')
        
model.load_state_dict(torch.load(modelname))


test(train_dataset, model, device, criterion, criterion1)
test(val_dataset, model, device, criterion, criterion1)
test(test_dataset, model, device, criterion, criterion1)



'''
[Test]: 1143it [00:03, 325.59it/s]
lossAverage: 0.09975876085427632 loss: 0.018110981211066246 MSE1: 0.0041492134 MAE1: 0.048976146 R21: 0.6545688216156549 MSE2: 0.0004266309 MAE2: 0.015676074 R22: 0.6527518798688585
[Test]: 143it [00:00, 308.85it/s]
lossAverage: 18.542519886978248 loss: 0.0311762485653162 MSE1: 0.005431875 MAE1: 0.04999218 R21: 0.5506259015330437 MSE2: 0.0003851711 MAE2: 0.014916257 R22: 0.6888732720596409
[Test]: 145it [00:00, 281.55it/s]
lossAverage: 0.1585997858047935 loss: 0.050016485154628754 MSE1: 0.0042366525 MAE1: 0.049788266 R21: 0.6781036425833643 MSE2: 0.0004365281 MAE2: 0.01599607 R22: 0.6740289343331869

[Test]: 1143it [00:03, 324.52it/s]
lossAverage: 0.11519754594266769 loss: 0.021116362884640694 MSE1: 0.004896068 MAE1: 0.052932855 R21: 0.631270000444443 MSE2: 0.00049608096 MAE2: 0.016871996 R22: 0.6269637851256782
[Test]: 143it [00:00, 283.73it/s]
lossAverage: 21.219560565298227 loss: 0.07730676233768463 MSE1: 0.006004407 MAE1: 0.051828377 R21: 0.548443911938516 MSE2: 0.00041701453 MAE2: 0.015657762 R22: 0.6870895974825267
[Test]: 145it [00:00, 264.12it/s]
lossAverage: 0.162769743280409 loss: 0.17872819304466248 MSE1: 0.0047573387 MAE1: 0.051048152 R21: 0.6857967137095158 MSE2: 0.000485776 MAE2: 0.016481854 R22: 0.6798371909383906

[Test]: 1143it [00:03, 323.74it/s]
lossAverage: 0.0923438661999835 loss: 0.025225277990102768 MSE1: 0.0039406684 MAE1: 0.04754775 R21: 0.6739719437539446 MSE2: 0.00040116478 MAE2: 0.015166105 R22: 0.6726640615283241
[Test]: 143it [00:00, 242.08it/s]
lossAverage: 18.931702666691454 loss: 0.023543821647763252 MSE1: 0.005367821 MAE1: 0.048827633 R21: 0.5262524152892094 MSE2: 0.00037365992 MAE2: 0.01472859 R22: 0.6731768370168081
[Test]: 145it [00:00, 251.05it/s]
lossAverage: 0.14536525803920586 loss: 0.045745570212602615 MSE1: 0.0041760756 MAE1: 0.048400603 R21: 0.695618219236751 MSE2: 0.00044117437 MAE2: 0.015766878 R22: 0.6833104883186959
'''

