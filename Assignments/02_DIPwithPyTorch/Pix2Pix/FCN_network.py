# import torch
import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),  # Input channels: 8, Output channels: 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # Input channels: 16, Output channels: 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Input channels: 16, Output channels: 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Input channels: 32, Output channels: 16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Input channels: 32, Output channels: 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 16, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 16, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),  # Input channels: 8, Output channels: 3
            nn.BatchNorm2d(3),
            # nn.ReLU(inplace=True)
            nn.Tanh()
        )
        ### None: since last layer outputs RGB channels, may need specific activation function

    def forward(self, x):
        # Encoder forward pass
        
        # Decoder forward pass
        code = self.conv1(x)
        output = self.conv2(code)
        ### FILL: encoder-decoder forward pass

        # output = ...
        
        return output
    

# if __name__ == '__main__':
#     net = FullyConvNetwork()
    
#     input = torch.randn(1, 3, 256, 256)
#     output = net(input)
#     print(output.shape)
