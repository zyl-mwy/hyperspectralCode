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
num_epochs = 100
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
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = rule)
    # CosineAnnealingLR 余弦退火调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,
                                                                eta_min=0, last_epoch=-1)
        
# 955 168
trainAutoEncoder = True
# 训练模型
if trainAutoEncoder:
    lossMin = 2
    R2Max = -200
    R21Max = -200
    R22Max = -200
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
            loss = criterion1(output, label.to(device)) + criterion(output, label.to(device))

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
        # https://www.zhihu.com/question/330027160
        '''
        均方根误差 (RMSE)：一般而言，取值越低越好，一般取值小于0.5为表现良好；
        均方误差 (MSE)：也是一般而言取值越低越好，一般取值小于0.2为表现良好；
        平均绝对误差（MAE）：一般取值小于0.1为表现良好；
        平均相对百分误差（MAPE）：一般取值小于10%为表现良好；
        决定系数R2：一般取值大于0.8为表现良好。
        '''
        # print(outputList)
        # MSE1 = mean_squared_error(outputList[:, 0], labelList[:, 0])
        # MAE1 = mean_absolute_error(outputList[:, 0], labelList[:, 0])
        R2 = r2_score(outputList, labelList) 
        R21 = r2_score(outputList[:, 0], labelList[:, 0])    
        # MSE2 = mean_squared_error(outputList[:, 1], labelList[:, 1])
        # MAE2 = mean_absolute_error(outputList[:, 1], labelList[:, 1])
        R22 = r2_score(outputList[:, 1], labelList[:, 1])
        # model.eval()
        # # 模型测试
        # lossTatol = 0
        # t = tqdm.tqdm(enumerate(val_loader),desc = f'[Val]') # ,loss:{loss.item()}
        # for step, (img, label) in t:
        #     # output = model(img.to(device))
        #     output = model(img.permute(0, 1, 4, 2, 3).to(device))
        #     # loss = criterion(output, label.to(device))
        #     # loss = criterion(output, label.to(device)) + criterion1(output, label.to(device))
        #     loss = criterion1(output, label.to(device)) + criterion(output, label.to(device))
        #     # print(img[0,0,0,0,0], output[0,0,0,0,0])
        #     lossTatol += loss.item()

        # lossAverage = lossTatol/(step+1)
        # if R22+R21 > R2Max:
        if R2 > R2Max:
            # R2Max = R22+R21
            R2Max = R2
            R21Max = R21
            R22Max = R22
            torch.save(model.state_dict(),modelname)
            print('Model Saved!! R2Max: {:.4f}, R21Max: {:.4f}, R22Max: {:.4f}'.format(R2Max, R21Max, R22Max))
        # 打印训练信息
        else:
            print('R2Max: {:.4f}, R21Max: {:.4f}, R22Max: {:.4f}'.format(R2Max, R21Max, R22Max))
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
[Test]: 1143it [00:03, 312.86it/s]
lossAverage: 0.08793002149158466 loss: 0.09984756261110306 MSE1: 0.0036518145 MAE1: 0.045688163 R21: 0.5960879945115947 MSE2: 0.00037408763 MAE2: 0.01463707 R22: 0.5864274547803137
[Test]: 143it [00:00, 253.55it/s]
lossAverage: 16.31237207635153 loss: 0.0054132966324687 MSE1: 0.0048149684 MAE1: 0.047562774 R21: 0.4578711311476088 MSE2: 0.00033225474 MAE2: 0.014188538 R22: 0.6270037064423126
[Test]: 145it [00:00, 297.13it/s]
lossAverage: 0.16250803400764458 loss: 0.013202476315200329 MSE1: 0.004287602 MAE1: 0.050333712 R21: 0.5813976190693606 MSE2: 0.00043559578 MAE2: 0.016043302 R22: 0.5730872612557469

[Test]: 1143it [00:03, 325.08it/s]
lossAverage: 0.07862721211682527 loss: 0.010687685571610928 MSE1: 0.0034591479 MAE1: 0.044280577 R21: 0.6342806714809275 MSE2: 0.00035396888 MAE2: 0.014162846 R22: 0.6229631366584396
[Test]: 143it [00:00, 274.18it/s]
lossAverage: 16.676836622922515 loss: 0.07302194833755493 MSE1: 0.00469409 MAE1: 0.045508496 R21: 0.4925426961908723 MSE2: 0.00031884064 MAE2: 0.013519555 R22: 0.6548799613016172
[Test]: 145it [00:00, 266.28it/s]
lossAverage: 0.13809440101729956 loss: 0.04647451639175415 MSE1: 0.004160926 MAE1: 0.050138876 R21: 0.5924977821684381 MSE2: 0.00042519008 MAE2: 0.016072178 R22: 0.5822408749838577

[Test]: 1143it [00:03, 333.96it/s]
lossAverage: 0.11446513128890067 loss: 0.0034603970125317574 MSE1: 0.0044992436 MAE1: 0.05118898 R21: 0.5541005542982703 MSE2: 0.00045425998 MAE2: 0.016260542 R22: 0.543017368430462
[Test]: 143it [00:00, 187.17it/s]
lossAverage: 17.42883210151959 loss: 0.04399528726935387 MSE1: 0.005262434 MAE1: 0.049088575 R21: 0.4681315395729081 MSE2: 0.0003712708 MAE2: 0.014737899 R22: 0.6204179888445446
[Test]: 145it [00:00, 245.35it/s]
lossAverage: 0.17531044800884485 loss: 0.03961997479200363 MSE1: 0.0045622913 MAE1: 0.05181423 R21: 0.6180213198056772 MSE2: 0.00046945046 MAE2: 0.016656207 R22: 0.6019647613813897 

[train]: 36it [00:01, 19.33it/s]
Epoch [99/100], AverageLoss: 0.0378, loss: 0.0284, lr: 2.467198171342e-07
[Test]: 143it [00:00, 330.26it/s]
R2Max: 0.6197, R21Max: 0.5506, R22Max: 0.6889

[train]: 36it [00:01, 19.42it/s]
Epoch [100/100], AverageLoss: 0.0378, loss: 0.0284, lr: 0.0
[Test]: 143it [00:00, 326.78it/s]
R2Max: 0.6197, R21Max: 0.5506, R22Max: 0.6889

[Test]: 1143it [00:03, 325.59it/s]
lossAverage: 0.09975876085427632 loss: 0.018110981211066246 MSE1: 0.0041492134 MAE1: 0.048976146 R21: 0.6545688216156549 MSE2: 0.0004266309 MAE2: 0.015676074 R22: 0.6527518798688585
[Test]: 143it [00:00, 308.85it/s]
lossAverage: 18.542519886978248 loss: 0.0311762485653162 MSE1: 0.005431875 MAE1: 0.04999218 R21: 0.5506259015330437 MSE2: 0.0003851711 MAE2: 0.014916257 R22: 0.6888732720596409
[Test]: 145it [00:00, 281.55it/s]
lossAverage: 0.1585997858047935 loss: 0.050016485154628754 MSE1: 0.0042366525 MAE1: 0.049788266 R21: 0.6781036425833643 MSE2: 0.0004365281 MAE2: 0.01599607 R22: 0.6740289343331869
'''

