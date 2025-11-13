import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Improved AlexNet-style CNN with ResNet concepts:
    - Same 5 convolutional layers as original
    - Added skip connections for better gradient flow
    - Batch normalization for stable training
    - Improved activation patterns
    """
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()

        # Conv Layer 1: 3 -> 64 channels (with BatchNorm)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Conv Layer 2: 64 -> 192 channels (with BatchNorm)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(192)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Conv Layer 3: 192 -> 384 channels (with BatchNorm)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(384)

        # Conv Layer 4: 384 -> 256 channels (with BatchNorm + Skip Connection)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        # Skip connection adapter (384 -> 256 channels)
        self.skip_conv4 = nn.Conv2d(384, 256, kernel_size=1, bias=False)
        self.skip_bn4 = nn.BatchNorm2d(256)

        # Conv Layer 5: 256 -> 256 channels (with BatchNorm + Skip Connection)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Improved classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 3 * 3, 1024),  # Reduced from 4096
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Conv Layer 1: 3 -> 64 channels
        x = F.relu(self.bn1(self.conv1(x)))  # 64x64 -> 32x32
        x = self.maxpool1(x)                 # 32x32 -> 15x15

        # Conv Layer 2: 64 -> 192 channels
        x = F.relu(self.bn2(self.conv2(x)))  # 15x15 -> 15x15
        x = self.maxpool2(x)                 # 15x15 -> 7x7

        # Conv Layer 3: 192 -> 384 channels
        x = F.relu(self.bn3(self.conv3(x)))  # 7x7 -> 7x7

        # Conv Layer 4: 384 -> 256 channels (WITH SKIP CONNECTION)
        identity = self.skip_bn4(self.skip_conv4(x))  # Adapt 384->256
        x = self.bn4(self.conv4(x))
        x = x + identity  # Skip connection
        x = F.relu(x)

        # Conv Layer 5: 256 -> 256 channels (WITH SKIP CONNECTION)
        identity = x  # Same dimensions, direct skip
        x = self.bn5(self.conv5(x))
        x = x + identity  # Skip connection
        x = F.relu(x)
        x = self.maxpool3(x)  # 7x7 -> 3x3

        # Classification
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
