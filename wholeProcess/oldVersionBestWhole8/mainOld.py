import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils


import tqdm

from dataSetLoad import Hyper_IMG
from autoEncoder import Autoencoder
from retrieve import HybridSN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = Hyper_IMG(r'E:\my_project\hyperspectralData\medician\leaf',train="Train",transform=transforms.ToTensor())
val_dataset = Hyper_IMG(r'E:\my_project\hyperspectralData\medician\leaf',train="Val",transform=transforms.ToTensor())
test_dataset = Hyper_IMG(r'E:\my_project\hyperspectralData\medician\leaf',train="Test",transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

model = Autoencoder()

modelname = 'autoEncoder.pth'
try:
    model.load_state_dict(torch.load(modelname))
    print('[INFO] Load Model complete')
except:
    pass

# 定义损失函数
criterion = nn.MSELoss()
# criterion = nn.BCELoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.002) # 0.001
# optimizer = optim.SGD(model.parameters(), lr=0.0001)


trainAutoEncoder = False
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
            output = model(img.to(device))
            loss = criterion(output, img.to(device))
            # print(loss.item())
            # break
            lossTatol += loss.item()
            # print(output[0,0,0,0,0], data[0,0,0,0,0])
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        lossAverage = lossTatol/(step+1)
        # if lossAverage < lossMin:
        #     lossMin = lossAverage
        #     torch.save(model.state_dict(),modelname)
        #     print('Epoch [{}/{}], Model Saved!! AverageLoss: {:.4f}, loss: {:.4f}, lossMin: {:.4f}'.format(epoch+1, num_epochs, lossAverage, loss.item(), lossMin))
        # # 打印训练信息
        # else:
        #     print('Epoch [{}/{}], AverageLoss: {:.4f}, loss: {:.4f}, lossMin: {:.4f}'.format(epoch+1, num_epochs, lossAverage, loss.item(), lossMin))
        print('Epoch [{}/{}], AverageLoss: {:.4f}, loss: {:.4f}'.format(epoch+1, num_epochs, lossAverage, loss.item()))

        model.eval()
        # 模型测试
        lossTatol = 0
        t = tqdm.tqdm(enumerate(val_loader),desc = f'[Val]') # ,loss:{loss.item()}
        for step, (img, label1, label2) in t:
            output = model(img.to(device))
            loss = criterion(output, img.to(device))
            print(img[0,0,0,0,0], output[0,0,0,0,0])
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

    model.eval()
    # 模型测试
    lossTatol = 0
    t = tqdm.tqdm(enumerate(test_loader),desc = f'[Test]') # ,loss:{loss.item()}
    for step, (img, label1, label2) in t:
        output = model(img)
        loss = criterion(output, img)
        
        lossTatol += loss.item()

    lossAverage = lossTatol/(step+1)

    print('lossAverage:', lossAverage, 'loss:', loss.item())

K = 15
model1 = HybridSN(rate=16, class_num=1, windowSize=16, K=K)
# print(data.shape)
data = torch.randn(1, 16, 16, K)
data1 = torch.randn([1, 1, 30, 25, 25])

# print(data.permute(0, 3, 1, 2).shape)
# print(model1(data1))
print(model1(torch.unsqueeze(data.permute(0, 3, 1, 2), dim=0)))



