import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms, utils
import os
import spectral 
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CIFAR10_IMG(Dataset):

    def __init__(self, root, train='Train', transform = None, target_transform=None):
        super(CIFAR10_IMG, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train=='Train':
            file_annotation = root + '/FinalTrain.xlsx'
            img_folder = root + '/drawRigion/maskPicNormal'
        elif self.train=='Val':
            file_annotation = root + '/FinalVal.xlsx'
            img_folder = root + '/drawRigion/maskPicNormal'
        elif self.train=='Test':
            file_annotation = root + '/FinalTest.xlsx'
            img_folder = root + '/drawRigion/maskPicNormal'
        myAnnotion = pd.read_excel(file_annotation).values
        # assert len(data_dict['images'])==len(data_dict['categories'])
        self.num_data = myAnnotion.shape[0]
        self.filenames = []
        self.labels1 = []
        self.labels2 = []
        self.img_folder = img_folder
        # print('---', self.img_folder)
        for i in range(self.num_data):
            self.filenames.append(myAnnotion[i][2])
            self.labels1.append(myAnnotion[i][8])
            self.labels2.append(myAnnotion[i][9])
        # print(self.filenames)
        # print(self.labels1)
        # print(self.labels2)

    def __getitem__(self, index):
        img_name = os.path.join(self.img_folder, self.filenames[index])
        label1 = self.labels1[index]
        label2 = self.labels2[index]
        img = spectral.envi.open(img_name+'.hdr', img_name+'.img').read_bands([i for i in range(204)])
        # print(img)
        # img = plt.imread(img_name)
        
        if self.transform is not None:
            img = torch.unsqueeze(self.transform(img).permute(1, 2, 0), dim=0) # 
        # print(img.shape)


        # return img, label
        # print(self.img_folder)
        return img, label1, label2#  + self.filenames[index]

    def __len__(self):
        return self.num_data
    
train_dataset = CIFAR10_IMG(r'E:\my_project\hyperspectralData\medician\leaf',train="Train",transform=transforms.ToTensor())
val_dataset = CIFAR10_IMG(r'E:\my_project\hyperspectralData\medician\leaf',train="Val",transform=transforms.ToTensor())
test_dataset = CIFAR10_IMG(r'E:\my_project\hyperspectralData\medician\leaf',train="Test",transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
# for step ,(b_x,b_y) in enumerate(train_loader):
# for step, (img, label1, label2) in enumerate(train_loader):
#     if step < 1:
#         print(img, label1, label2)
    #     imgs = utils.make_grid(b_x)
    #     print(imgs.shape)
    #     imgs = np.transpose(imgs,(1,2,0))
    #     print(imgs.shape)
    #     plt.imshow(imgs)
    #     plt.show()

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器层
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # 解码器层
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class Autoencoder1(nn.Module):
    def __init__(self):
        super(Autoencoder1, self).__init__()
        # 编码器层
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=[3, 3, 5], stride=1, padding=[1, 1, 0]),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=[1, 1, 2], stride=[1, 1, 2]),
            nn.Conv3d(16, 32, kernel_size=[3, 3, 21], stride=1, padding=[1, 1, 0]),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=[1, 1, 2], stride=[1, 1, 2]),
            nn.Conv3d(32, 64, kernel_size=[3, 3, 21], stride=1, padding=[1, 1, 0]),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=[1, 1, 2], stride=[1, 1, 2])
        )
        
        # 解码器层
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=[1, 1, 21], stride=[1, 1, 2], padding=[0, 0, 0], output_padding=[0, 0, 1]),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, kernel_size=[1, 1, 21], stride=[1, 1, 2], padding=[0, 0, 0], output_padding=[0, 0, 1]),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 1, kernel_size=[1, 1, 5], stride=[1, 1, 2], padding=[0, 0, 0], output_padding=[0, 0, 1]),
            nn.Sigmoid()
            # nn.ReLU(True)
            # nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 初始化模型
model1 = Autoencoder()
model = Autoencoder1()

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

# 加载高光谱数据
# data = torch.randn(1, 1, 16, 16, 204)
data = torch.randn(1, 16, 16, 204)
# print(data.shape, model1(data).shape)

# 训练模型
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
compressed_data = model.cpu().encoder(data)
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


