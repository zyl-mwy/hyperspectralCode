import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms, utils
import os
import spectral 
import tqdm
import torch.nn.functional as F

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
    
rate = 16
class_num = 1
windowSize = 25
K = 30
    
class HybridSN(nn.Module):
  #定义各个层的部分
    def __init__(self):
        super(HybridSN, self).__init__()
        self.S = windowSize
        self.L = K;

        #self.conv_block = nn.Sequential()
        ## convolutional layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3))
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3))
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))
        
        #不懂 inputX经过三重3d卷积的大小
        inputX = self.get2Dinput()
        inputConv4 = inputX.shape[1] * inputX.shape[2]
        # conv4 （24*24=576, 19, 19），64个 3x3 的卷积核 ==>（（64, 17, 17）
        self.conv4 = nn.Conv2d(inputConv4, 64, kernel_size=(3, 3))

        #self-attention
        self.sa1 = nn.Conv2d(64, 64//rate, kernel_size=1)
        self.sa2 = nn.Conv2d(64//rate, 64, kernel_size=1)
        
        # 全连接层（256个节点） # 64 * 17 * 17 = 18496
        self.dense1 = nn.Linear(18496, 256) 
        # 全连接层（128个节点）
        self.dense2 = nn.Linear(256, 128)
        # 最终输出层(16个节点)
        self.dense3 = nn.Linear(128, class_num)
        
        #让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
        #但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了
        #参考: https://blog.csdn.net/yangfengling1023/article/details/82911306
        #self.drop = nn.Dropout(p = 0.4)
        #改成0.43试试
        self.drop = nn.Dropout(p = 0.43)
        self.soft = nn.Softmax(dim=1)
        pass

  #辅助函数，没怎么懂，求经历过三重卷积后二维的一个大小
    def get2Dinput(self):
        #torch.no_grad(): 做运算，但不计入梯度记录
        with torch.no_grad():
            x = torch.zeros((1, 1, self.L, self.S, self.S))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x
        pass

  #必须重载的部分，X代表输入
    def forward(self, x):
        print(x.shape)
        #F在上文有定义torch.nn.functional，是已定义好的一组名称
        # print(x.shape)
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        # 进行二维卷积，因此把前面的 32*18 reshape 一下，得到 （576, 19, 19）
        out = out.view(-1, out.shape[1] * out.shape[2], out.shape[3], out.shape[4])
        out = F.relu(self.conv4(out))

        # Squeeze 第三维卷成1了
        weight = F.avg_pool2d(out, out.size(2))    #参数为输入，kernel
        #参考: https://blog.csdn.net/qq_21210467/article/details/81415300
        #参考: https://blog.csdn.net/u013066730/article/details/102553073

        # Excitation: sa（压缩到16分之一）--Relu--fc（激到之前维度）--Sigmoid（保证输出为0至1之间）attention
        weight = F.relu(self.sa1(weight))
        weight = F.sigmoid(self.sa2(weight))
        out = out * weight

        # flatten: 变为 18496 维的向量，
        out = out.view(out.size(0), -1)

        out = F.relu(self.dense1(out))
        out = self.drop(out)
        out = F.relu(self.dense2(out))
        out = self.drop(out)
        out = self.dense3(out)

        #添加此语句后出现LOSS不下降的情况，参考：https://www.e-learn.cn/topic/3733809
        #原因是CrossEntropyLoss()=softmax+负对数损失（已经包含了softmax)。如果多写一次softmax，则结果会发生错误
        #out = self.soft(out)
        #out = F.log_softmax(out)

        return out

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

model2 = HybridSN()
# print(data.shape)
data1 = torch.randn([1, 1, 30, 25, 25])
# print(data.permute(0, 3, 1, 2).shape)
print(model2(data1))
# print(model2(torch.unsqueeze(data.permute(0, 3, 1, 2), dim=0)))



