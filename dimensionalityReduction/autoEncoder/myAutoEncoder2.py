import torch
import torch.nn as nn

# 定义3D自编码器模型
class Autoencoder3D(nn.Module):
    def __init__(self):
        super(Autoencoder3D, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 加载数据
data = torch.randn(1, 1, 16, 16, 204)  # 输入数据的维度为[batch_size, channels, depth, height, width]

# 初始化模型
model = Autoencoder3D()
# print(model.parameters())
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    # 前向传播
    output = model(data)
    print(data.shape, output.shape)
    loss = criterion(output, data)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失值
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 保存模型
torch.save(model.state_dict(), 'autoencoder3d_model.pth')
