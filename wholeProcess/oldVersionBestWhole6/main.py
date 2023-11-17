import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils


import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataSetLoad import Hyper_IMG
from autoEncoder import Autoencoder
from retrieve import HybridSN

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
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
            loss = criterion1(output, label.to(device))

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
            # loss = criterion(output, label.to(device)) + criterion1(output, label.to(device))
            loss = criterion1(output, label.to(device))
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
model = model.to(device)

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
# plt.plot(outputList)
# plt.plot(labelList)

plt.plot([i/100 for i in range(100)], [i/100 for i in range(100)])
plt.plot([i/100 for i in range(100)], [i/100+0.05 for i in range(100)])
plt.plot([i/100 for i in range(100)], [i/100-0.05 for i in range(100)])
plt.scatter(outputList[:, 0], labelList[:, 0])
plt.show()
plt.plot([i/100 for i in range(100)], [i/100 for i in range(100)])
plt.plot([i/100 for i in range(100)], [i/100+0.05 for i in range(100)])
plt.plot([i/100 for i in range(100)], [i/100-0.05 for i in range(100)])
plt.scatter(outputList[:, 1], labelList[:, 1])
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
plt.plot([i/100 for i in range(100)], [i/100 for i in range(100)])
plt.plot([i/100 for i in range(100)], [i/100+0.05 for i in range(100)])
plt.plot([i/100 for i in range(100)], [i/100-0.05 for i in range(100)])
plt.scatter(outputList[:, 0], labelList[:, 0])
plt.show()
plt.plot([i/100 for i in range(100)], [i/100 for i in range(100)])
plt.plot([i/100 for i in range(100)], [i/100+0.05 for i in range(100)])
plt.plot([i/100 for i in range(100)], [i/100-0.05 for i in range(100)])
plt.scatter(outputList[:, 1], labelList[:, 1])
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
plt.plot([i/100 for i in range(100)], [i/100 for i in range(100)])
plt.plot([i/100 for i in range(100)], [i/100+0.05 for i in range(100)])
plt.plot([i/100 for i in range(100)], [i/100-0.05 for i in range(100)])
plt.scatter(outputList[:, 0], labelList[:, 0])
plt.show()
plt.plot([i/100 for i in range(100)], [i/100 for i in range(100)])
plt.plot([i/100 for i in range(100)], [i/100+0.05 for i in range(100)])
plt.plot([i/100 for i in range(100)], [i/100-0.05 for i in range(100)])
plt.scatter(outputList[:, 1], labelList[:, 1])
plt.show()


# print(data.shape)
# data = torch.randn(1, 16, 16, 204)
# data1 = torch.randn([1, 1, 30, 25, 25])

# print(data.permute(0, 3, 1, 2).shape)
# print(model1(data1))
# print(model1(torch.unsqueeze(data.permute(0, 3, 1, 2), dim=0)))

'''
[Test]: 1143it [00:03, 346.66it/s]
lossAverage: 0.10562578111962319 loss: 0.16147424280643463 MSE1: 0.0040696864 MAE1: 0.049032997 R21: 0.4728264177306105 MSE2: 0.002955161 MAE2: 0.017327135 R22: -3.127296691660767
[Test]: 143it [00:00, 272.38it/s]
lossAverage: 16.223807831647946 loss: 0.017581304535269737 MSE1: 0.004795447 MAE1: 0.048011668 R21: 0.4030885109160218 MSE2: 0.0003422316 MAE2: 0.0145420125 R22: 0.5433295158944231
[Test]: 145it [00:00, 232.37it/s]
lossAverage: 0.16702118115579903 loss: 0.011986001394689083 MSE1: 0.004339824 MAE1: 0.050636083 R21: 0.5107692786027586 MSE2: 0.00045198377 MAE2: 0.01634121 R22: 0.45055652664042967

[Test]: 1143it [00:03, 349.53it/s]
lossAverage: 0.08304427414367584 loss: 0.0009593871072866023 MSE1: 0.003506698 MAE1: 0.04480332 R21: 0.6362974379073663 MSE2: 0.0029231303 MAE2: 0.015840122 R22: -2.3335208507544025
[Test]: 143it [00:00, 308.86it/s]
lossAverage: 17.45443259337816 loss: 0.07494880259037018 MSE1: 0.005024462 MAE1: 0.049134508 R21: 0.4798216672965756 MSE2: 0.00033918925 MAE2: 0.014595321 R22: 0.616865877654602
[Test]: 145it [00:00, 293.52it/s]
lossAverage: 0.1832724545378902 loss: 0.0054364814423024654 MSE1: 0.0047430694 MAE1: 0.05188116 R21: 0.5524049733133559 MSE2: 0.00050009316 MAE2: 0.016597148 R22: 0.47987483677650344

[Test]: 1143it [00:03, 313.48it/s]
lossAverage: 0.09074329081357428 loss: 0.006631555035710335 MSE1: 0.0037185731 MAE1: 0.046512295 R21: 0.5893470350516089 MSE2: 0.0028921897 MAE2: 0.01657856 R22: -2.4552047875182255
[Test]: 143it [00:00, 280.94it/s]
lossAverage: 15.44789004382436 loss: 0.002765433397144079 MSE1: 0.0050506582 MAE1: 0.04836199 R21: 0.42875267821509766 MSE2: 0.00037619207 MAE2: 0.01481283 R22: 0.5378705880265425     
[Test]: 145it [00:00, 258.93it/s]
lossAverage: 0.14956270866090296 loss: 0.030909186229109764 MSE1: 0.004551806 MAE1: 0.05216749 R21: 0.5848144801140709 MSE2: 0.00046659246 MAE2: 0.016751453 R22: 0.5419177309390999

[Test]: 1143it [00:03, 292.02it/s]
lossAverage: 0.07792053128400628 loss: 0.08751692622900009 MSE1: 0.0032381616 MAE1: 0.04329281 R21: 0.6593109458865165 MSE2: 0.0028830925 MAE2: 0.015738558 R22: -2.3704963475721903    
[Test]: 143it [00:00, 287.73it/s]
lossAverage: 16.55541319127356 loss: 0.01004988793283701 MSE1: 0.004913671 MAE1: 0.047767993 R21: 0.44741690081231744 MSE2: 0.0003651443 MAE2: 0.014654396 R22: 0.5370476001176863      
[Test]: 145it [00:00, 290.00it/s]
lossAverage: 0.12740825169708717 loss: 0.004174989182502031 MSE1: 0.004162579 MAE1: 0.05072743 R21: 0.6339016468188059 MSE2: 0.00042281402 MAE2: 0.016148735 R22: 0.5932948354592295 

[Test]: 1143it [00:03, 300.78it/s]
lossAverage: 0.08189901165481651 loss: 0.0932406410574913 MSE1: 0.003367837 MAE1: 0.04375618 R21: 0.640818340264652 MSE2: 0.002849575 MAE2: 0.01594103 R22: -2.5197521649087244
[Test]: 143it [00:00, 223.24it/s]
lossAverage: 15.936905074163173 loss: 0.0021224217489361763 MSE1: 0.0047245254 MAE1: 0.046632927 R21: 0.4775349015179007 MSE2: 0.0003410209 MAE2: 0.014293039 R22: 0.562560377841796
[Test]: 145it [00:00, 266.06it/s]
lossAverage: 0.16380021086469837 loss: 0.0014733123825863004 MSE1: 0.004476886 MAE1: 0.05113423 R21: 0.5654051443949042 MSE2: 0.0004785097 MAE2: 0.016480634 R22: 0.4737057970169096

[Test]: 1143it [00:03, 292.02it/s]
lossAverage: 0.07792053128400628 loss: 0.08751692622900009 MSE1: 0.0032381616 MAE1: 0.04329281 R21: 0.6593109458865165 MSE2: 0.0028830925 MAE2: 0.015738558 R22: -2.3704963475721903    
[Test]: 143it [00:00, 287.73it/s]
lossAverage: 16.55541319127356 loss: 0.01004988793283701 MSE1: 0.004913671 MAE1: 0.047767993 R21: 0.44741690081231744 MSE2: 0.0003651443 MAE2: 0.014654396 R22: 0.5370476001176863      
[Test]: 145it [00:00, 290.00it/s]
lossAverage: 0.12740825169708717 loss: 0.004174989182502031 MSE1: 0.004162579 MAE1: 0.05072743 R21: 0.6339016468188059 MSE2: 0.00042281402 MAE2: 0.016148735 R22: 0.5932948354592295 
'''

