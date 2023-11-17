import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),  # 16x16x16
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 8x8x8
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),  # 8x8x8
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)  # 4x4x4
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),  # 8x8x8
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=2, stride=2),  # 16x16x16
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded