import torch.nn as nn

# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         # 编码器层
#         self.encoder = nn.Sequential(
#             nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool3d(kernel_size=2, stride=2),
#             nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool3d(kernel_size=2, stride=2)
#         )
        
#         # 解码器层
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
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