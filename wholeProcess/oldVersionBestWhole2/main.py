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
model = HybridSN(rate=16, class_num=2, windowSize=16, K=204)

modelname = 'main.pth'
# try:
#     model.load_state_dict(torch.load(modelname))
#     print('[INFO] Load Model complete')
# except:
#     pass

# 定义损失函数
class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()
    def forward(self, y_pred, y_true):
        y_mean = torch.mean(y_true)
        ss_lot = torch.sum((y_true - y_mean) ** 2)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        return ss_res / ss_lot
criterion = nn.MSELoss()
# criterion = nn.MSELoss() + R2Loss()
criterion1 = R2Loss()
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
        for step, (img, label) in t:
            # if step < 1:
            #     print(img.shape, label1.shape, label2.shape)
        # 前向传播
            # print(step)
            # print(img.shape)
            output = model(img.permute(0, 1, 4, 2, 3).to(device))
            # print(output.shape)
            # print(label.shape)
            # loss = criterion(output, label.to(device))
            loss = criterion(output, label.to(device)) + criterion1(output, label.to(device))

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
        for step, (img, label) in t:
            # output = model(img.to(device))
            output = model(img.permute(0, 1, 4, 2, 3).to(device))
            # loss = criterion(output, label.to(device))
            loss = criterion(output, label.to(device)) + criterion1(output, label.to(device))
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
    # https://www.zhihu.com/question/330027160
    '''
    均方根误差 (RMSE)：一般而言，取值越低越好，一般取值小于0.5为表现良好；
    均方误差 (MSE)：也是一般而言取值越低越好，一般取值小于0.2为表现良好；
    平均绝对误差（MAE）：一般取值小于0.1为表现良好；
    平均相对百分误差（MAPE）：一般取值小于10%为表现良好；
    决定系数R2：一般取值大于0.8为表现良好。
    '''
    # print(outputList)
    MSE1 = mean_squared_error(outputList[:, 0], labelList[:, 0])
    MAE1 = mean_absolute_error(outputList[:, 0], labelList[:, 0])
    R21 = r2_score(outputList[:, 0], labelList[:, 0])    
    MSE2 = mean_squared_error(outputList[:, 1], labelList[:, 1])
    MAE2 = mean_absolute_error(outputList[:, 1], labelList[:, 1])
    R22 = r2_score(outputList[:, 1], labelList[:, 1])  
    print('lossAverage:', lossAverage, 'loss:', loss.item(), 'MSE1:', MSE1, 'MAE1:', MAE1, 'R21:', R21, 'MSE2:', MSE2, 'MAE2:', MAE2, 'R22:', R22)
    # print(outputList)
    # print(labelList)
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
    # https://www.zhihu.com/question/330027160
    '''
    均方根误差 (RMSE)：一般而言，取值越低越好，一般取值小于0.5为表现良好；
    均方误差 (MSE)：也是一般而言取值越低越好，一般取值小于0.2为表现良好；
    平均绝对误差（MAE）：一般取值小于0.1为表现良好；
    平均相对百分误差（MAPE）：一般取值小于10%为表现良好；
    决定系数R2：一般取值大于0.8为表现良好。
    '''
    # print(outputList)
    MSE1 = mean_squared_error(outputList[:, 0], labelList[:, 0])
    MAE1 = mean_absolute_error(outputList[:, 0], labelList[:, 0])
    R21 = r2_score(outputList[:, 0], labelList[:, 0])    
    MSE2 = mean_squared_error(outputList[:, 1], labelList[:, 1])
    MAE2 = mean_absolute_error(outputList[:, 1], labelList[:, 1])
    R22 = r2_score(outputList[:, 1], labelList[:, 1])  
    print('lossAverage:', lossAverage, 'loss:', loss.item(), 'MSE1:', MSE1, 'MAE1:', MAE1, 'R21:', R21, 'MSE2:', MSE2, 'MAE2:', MAE2, 'R22:', R22)
    # print(outputList)
    # print(labelList)
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
    # https://www.zhihu.com/question/330027160
    '''
    均方根误差 (RMSE)：一般而言，取值越低越好，一般取值小于0.5为表现良好；
    均方误差 (MSE)：也是一般而言取值越低越好，一般取值小于0.2为表现良好；
    平均绝对误差（MAE）：一般取值小于0.1为表现良好；
    平均相对百分误差（MAPE）：一般取值小于10%为表现良好；
    决定系数R2：一般取值大于0.8为表现良好。
    '''
    # print(outputList)
    MSE1 = mean_squared_error(outputList[:, 0], labelList[:, 0])
    MAE1 = mean_absolute_error(outputList[:, 0], labelList[:, 0])
    R21 = r2_score(outputList[:, 0], labelList[:, 0])    
    MSE2 = mean_squared_error(outputList[:, 1], labelList[:, 1])
    MAE2 = mean_absolute_error(outputList[:, 1], labelList[:, 1])
    R22 = r2_score(outputList[:, 1], labelList[:, 1])  
    print('lossAverage:', lossAverage, 'loss:', loss.item(), 'MSE1:', MSE1, 'MAE1:', MAE1, 'R21:', R21, 'MSE2:', MSE2, 'MAE2:', MAE2, 'R22:', R22)
    # print(outputList)
    # print(labelList)
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
'''
lossAverage: 0.005376964797328029 loss: 0.015848642215132713 MSE: 0.005376964817658023 MAE: 0.05566091995852513 R2: 0.3148198491126709
[Test]: 143it [00:00, 322.80it/s]
lossAverage: 0.005323958141253162 loss: 0.0017158668488264084 MSE: 0.0053239581437367145 MAE: 0.05018107478435223 R2: 0.35734863874242173
[Test]: 145it [00:00, 363.41it/s]
lossAverage: 0.005044517306320641 loss: 4.457081013242714e-05 MSE: 0.005044517398377916 MAE: 0.053048982599685934 R2: 0.4856261108527372
'''

'''
[Test]: 1143it [00:03, 377.71it/s]
lossAverage: 0.0035812513497582 loss: 0.002472693333402276 MSE1: 0.004165897 MAE1: 0.049960833 R21: 0.40785007996777534 MSE2: 0.0029966058 MAE2: 0.018069327 R22: -4.265916323072954
[Test]: 143it [00:00, 210.29it/s]
lossAverage: 0.002916062903526179 loss: 0.00047163316048681736 MSE1: 0.00541964 MAE1: 0.05026748 R21: 0.3000875693727455 MSE2: 0.00041248603 MAE2: 0.015618823 R22: 0.34151220066473487
[Test]: 145it [00:00, 293.52it/s]
lossAverage: 0.002644353197396852 loss: 0.00224885530769825 MSE1: 0.004778289 MAE1: 0.05303728 R21: 0.4103182455797898 MSE2: 0.00051041757 MAE2: 0.017226394 R22: 0.23705808852657317
'''

'''
[Test]: 1143it [00:03, 355.36it/s]
lossAverage: 0.09808324227645603 loss: 0.2158178985118866 MSE1: 0.0038780419 MAE1: 0.04732175 R21: 0.5360724958508214 MSE2: 0.0029200425 MAE2: 0.017328305 R22: -3.503818988815522
[Test]: 143it [00:00, 234.21it/s]
lossAverage: 17.339904136377733 loss: 0.002167455153539777 MSE1: 0.0055648983 MAE1: 0.049746662 R21: 0.31941162117624533 MSE2: 0.00042105676 MAE2: 0.015581574 R22: 0.3512054432871192
[Test]: 145it [00:00, 197.82it/s]
lossAverage: 0.18128537986963708 loss: 0.010014357045292854 MSE1: 0.00442624 MAE1: 0.05065167 R21: 0.5551238836703907 MSE2: 0.00047042314 MAE2: 0.016794996 R22: 0.3992604063623765
'''

