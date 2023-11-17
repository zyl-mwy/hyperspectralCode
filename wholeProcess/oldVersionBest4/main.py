import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils


import tqdm
# import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataSetLoad import Hyper_IMG
from autoEncoder import Autoencoder
from retrieve import HybridSN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = Hyper_IMG(r'E:\my_project\hyperspectralData\medician\leaf',train="Train",transform=transforms.Compose([transforms.ToTensor()
                                                                                                                      # ,transforms.RandomHorizontalFlip(p=0.5)
                                                                                                                      # ,transforms.RandomVerticalFlip(p=0.5)
                                                                                                                      ])) # , transforms.Normalize(0.5, 0.5, 0)
val_dataset = Hyper_IMG(r'E:\my_project\hyperspectralData\medician\leaf',train="Val",transform=transforms.ToTensor())
test_dataset = Hyper_IMG(r'E:\my_project\hyperspectralData\medician\leaf',train="Test",transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True) # 32
val_loader = DataLoader(dataset=val_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

# model = Autoencoder()
model = HybridSN(rate=16, class_num=1, windowSize=16, K=204)

modelname = 'main.pth'
# try:
#     model.load_state_dict(torch.load(modelname))
#     print('[INFO] Load Model complete')
# except:
#     pass

# 定义损失函数
criterion = nn.MSELoss()
# criterion = nn.BCELoss()

'''trainLog
1—>0.00001 0.0133
128->0.001 0.0126
128->0.1 0.0124
'''
# 定义优化器
# optimizer = optim.Adam(model.parameters(), lr=0.001) # 
# optimizer = optim.SGD(model.parameters(), lr=0.00000001)
if True:
    # lambda1 = lambda epoch:np.sin(epoch) / epoch
    import math
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': 0.001}], lr=0.001) # 0.000001 0.001 , weight_decay=1e-6
    def rule(epoch):
        lamda = math.pow(0.1, epoch // 50)
        return lamda
    # print(rule(3))
    # print(rule(10))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = rule)
    # CosineAnnealingLR 余弦退火调整学习率
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100,
    #                                                             eta_min=0, last_epoch=-1)
        

trainAutoEncoder = True
# 训练模型
if trainAutoEncoder:
    num_epochs = 100
    lossMin = 2
    for epoch in range(num_epochs):

        model.train()
        model.to(device)
        lossTatol = 0
        t = tqdm.tqdm(enumerate(train_loader),desc = f'[train]') # ,loss:{loss.item()}
        for step, (img, label1, label2) in t:
            # if step < 1:
            #     print(img.shape, label1.shape, label2.shape)
        # 前向传播
            # print(step)
            # print(img.shape)
            output = model(img.permute(0, 1, 4, 2, 3).to(device))
            loss = criterion(output, label1.to(device))

            # regularization_loss = 0
            # for param in model.parameters():
            #     regularization_loss += torch.sum(abs(param))
            # loss = criterion(output, label1.to(device)) + 1e-6 * regularization_loss
            
            # print(img)
            # print(output[0])
            # print(loss.item())
            # break
            lossTatol += loss.item()
            # print(output[0,0,0,0,0], data[0,0,0,0,0])
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
            
            
        lossAverage = lossTatol/(step+1)
        # if lossAverage < lossMin:
        #     lossMin = lossAverage
        #     torch.save(model.state_dict(),modelname)
        #     print('Epoch [{}/{}], Model Saved!! AverageLoss: {:.4f}, loss: {:.4f}, lossMin: {:.4f}'.format(epoch+1, num_epochs, lossAverage, loss.item(), lossMin))
        # # 打印训练信息
        # else:
        #     print('Epoch [{}/{}], AverageLoss: {:.4f}, loss: {:.4f}, lossMin: {:.4f}'.format(epoch+1, num_epochs, lossAverage, loss.item(), lossMin))
        print('Epoch [{}/{}], AverageLoss: {:.4f}, loss: {:.4f}, lr: {}'.format(epoch+1, num_epochs, lossAverage, loss.item(), optimizer.state_dict()['param_groups'][0]['lr'])) # optimizer.state_dict()['param_groups'][0]['lr'])

        model.eval()
        # 模型测试
        lossTatol = 0
        t = tqdm.tqdm(enumerate(val_loader),desc = f'[Val]') # ,loss:{loss.item()}
        for step, (img, label1, label2) in t:
            # output = model(img.to(device))
            output = model(img.permute(0, 1, 4, 2, 3).to(device))
            loss = criterion(output, label1.to(device))
            # print(img[0,0,0,0,0], output[0,0,0,0,0])
            lossTatol += loss.item()

        lossAverage = lossTatol/(step+1)
        if lossAverage < lossMin:
            lossMin = lossAverage
            torch.save(model.state_dict(),modelname)
            print('Model Saved!! AverageLoss: {:.4f}, loss: {:.4f}, lossMin: {:.4f}'.format(lossAverage, loss.item(), lossMin))
        # 打印训练信息
        else:
            print('AverageLoss: {:.4f}, loss: {:.4f}, lossMin: {:.4f}'.format(lossAverage, loss.item(), lossMin))
        # print(lossAverage, loss.item())
        print(' ')
    # 获取降维后的数据
    # compressed_data = model.cpu().encoder(data)
    # print(compressed_data.shape)
    # compressed_data = compressed_data.view(-1, 16, 16, 10)
    # print('Compressed data shape:', compressed_data.shape)
    model.load_state_dict(torch.load(modelname))


    test_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    outputList = []
    labelList = []
    model.eval()
    # 模型测试
    lossTatol = 0
    t = tqdm.tqdm(enumerate(test_loader),desc = f'[Test]') # ,loss:{loss.item()}
    for step, (img, label1, label2) in t:
        output = model(img.permute(0, 1, 4, 2, 3).to(device))
        loss = criterion(output, label1.to(device))
        
        lossTatol += loss.item()
        outputList.append(output.item())
        labelList.append(label1.item())

    lossAverage = lossTatol/(step+1)
    # https://www.zhihu.com/question/330027160
    '''
    均方根误差 (RMSE)：一般而言，取值越低越好，一般取值小于0.5为表现良好；
    均方误差 (MSE)：也是一般而言取值越低越好，一般取值小于0.2为表现良好；
    平均绝对误差（MAE）：一般取值小于0.1为表现良好；
    平均相对百分误差（MAPE）：一般取值小于10%为表现良好；
    决定系数R2：一般取值大于0.8为表现良好。
    '''
    MSE = mean_squared_error(outputList, labelList)
    MAE = mean_absolute_error(outputList, labelList)
    R2 = r2_score(outputList, labelList)    
    print('lossAverage:', lossAverage, 'loss:', loss.item(), 'MSE:', MSE, 'MAE:', MAE, 'R2:', R2)
    # print(outputList)
    # print(labelList)
    import matplotlib.pyplot as plt
    plt.plot(outputList)
    plt.plot(labelList)
    plt.show()


    test_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
    outputList = []
    labelList = []
    model.eval()
    # 模型测试
    lossTatol = 0
    t = tqdm.tqdm(enumerate(test_loader),desc = f'[Test]') # ,loss:{loss.item()}
    for step, (img, label1, label2) in t:
        output = model(img.permute(0, 1, 4, 2, 3).to(device))
        loss = criterion(output, label1.to(device))
        
        lossTatol += loss.item()
        outputList.append(output.item())
        labelList.append(label1.item())

    lossAverage = lossTatol/(step+1)
    # https://www.zhihu.com/question/330027160
    '''
    均方根误差 (RMSE)：一般而言，取值越低越好，一般取值小于0.5为表现良好；
    均方误差 (MSE)：也是一般而言取值越低越好，一般取值小于0.2为表现良好；
    平均绝对误差（MAE）：一般取值小于0.1为表现良好；
    平均相对百分误差（MAPE）：一般取值小于10%为表现良好；
    决定系数R2：一般取值大于0.8为表现良好。
    '''
    MSE = mean_squared_error(outputList, labelList)
    MAE = mean_absolute_error(outputList, labelList)
    R2 = r2_score(outputList, labelList)    
    print('lossAverage:', lossAverage, 'loss:', loss.item(), 'MSE:', MSE, 'MAE:', MAE, 'R2:', R2)
    # print(outputList)
    # print(labelList)
    import matplotlib.pyplot as plt
    plt.plot(outputList)
    plt.plot(labelList)
    plt.show()


    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    outputList = []
    labelList = []
    model.eval()
    # 模型测试
    lossTatol = 0
    t = tqdm.tqdm(enumerate(test_loader),desc = f'[Test]') # ,loss:{loss.item()}
    for step, (img, label1, label2) in t:
        output = model(img.permute(0, 1, 4, 2, 3).to(device))
        loss = criterion(output, label1.to(device))
        
        lossTatol += loss.item()
        outputList.append(output.item())
        labelList.append(label1.item())

    lossAverage = lossTatol/(step+1)
    # https://www.zhihu.com/question/330027160
    '''
    均方根误差 (RMSE)：一般而言，取值越低越好，一般取值小于0.5为表现良好；
    均方误差 (MSE)：也是一般而言取值越低越好，一般取值小于0.2为表现良好；
    平均绝对误差（MAE）：一般取值小于0.1为表现良好；
    平均相对百分误差（MAPE）：一般取值小于10%为表现良好；
    决定系数R2：一般取值大于0.8为表现良好。
    '''
    MSE = mean_squared_error(outputList, labelList)
    MAE = mean_absolute_error(outputList, labelList)
    R2 = r2_score(outputList, labelList)    
    print('lossAverage:', lossAverage, 'loss:', loss.item(), 'MSE:', MSE, 'MAE:', MAE, 'R2:', R2)
    # print(outputList)
    # print(labelList)
    import matplotlib.pyplot as plt
    plt.plot(outputList)
    plt.plot(labelList)
    plt.show()


# print(data.shape)
# data = torch.randn(1, 16, 16, 204)
# data1 = torch.randn([1, 1, 30, 25, 25])

# print(data.permute(0, 3, 1, 2).shape)
# print(model1(data1))
# print(model1(torch.unsqueeze(data.permute(0, 3, 1, 2), dim=0)))

'''
[Test]: 1143it [00:02, 385.07it/s]
lossAverage: 0.0059283425277902 loss: 0.03199363127350807 MSE: 0.005928342512282992 MAE: 0.058735905304772754 R2: 0.16872749447772661
[Test]: 143it [00:00, 360.20it/s]
lossAverage: 0.005764150097272081 loss: 0.00041169446194544435 MSE: 0.005764150074299005 MAE: 0.05467507680812916 R2: 0.1891659446310372
[Test]: 145it [00:00, 326.57it/s]
lossAverage: 0.005263409113289809 loss: 0.0003608684055507183 MSE: 0.0052634092068072 MAE: 0.05366555113216926 R2: 0.3530620518114739
'''

