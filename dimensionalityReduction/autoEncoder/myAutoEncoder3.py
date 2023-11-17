import torch
import torch.nn as nn
import torch.optim as optim

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


# 定义损失函数
criterion = nn.MSELoss()
# criterion = nn.BCELoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.0001)

# 加载高光谱数据
data = torch.randn(1, 1, 16, 16, 204)
# print(data.shape, model1(data).shape)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    output = model(data)
    loss = criterion(output, data)
    # print(output[0,0,0,0,0], data[0,0,0,0,0])
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 打印训练信息
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 获取降维后的数据
compressed_data = model.encoder(data)
print(compressed_data.shape)
compressed_data = compressed_data.view(-1, 16, 16, 10)
print('Compressed data shape:', compressed_data.shape)