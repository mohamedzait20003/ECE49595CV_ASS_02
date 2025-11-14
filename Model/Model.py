import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Deep CNN with ResNet concepts and Bottleneck layer:
    - 10 convolutional layers total (5 original + 5 new)
    - Bottleneck block with skip connection for efficiency
    - Multiple skip connections for better gradient flow
    - Batch normalization for stable training
    - Progressive channel expansion
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

        # NEW: Conv Layer 6: 256 -> 384 channels
        self.conv6 = nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(384)

        # NEW: Conv Layer 7: 384 -> 384 channels (with Skip Connection)
        self.conv7 = nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(384)

        # NEW: Conv Layer 8: 384 -> 512 channels
        self.conv8 = nn.Conv2d(384, 512, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # NEW: Conv Layer 9: 512 -> 512 channels (with Skip Connection)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(512)

        # NEW: Conv Layer 10: 512 -> 512 channels (with Skip Connection)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(512)

        # BOTTLENECK BLOCK with Skip Connection
        # Reduces dimensionality: 512 -> 256 -> 512
        # 1x1 reduce
        self.bottleneck_conv1 = nn.Conv2d(512, 256, kernel_size=1,
                                          bias=False)
        self.bottleneck_bn1 = nn.BatchNorm2d(256)

        # 3x3 process
        self.bottleneck_conv2 = nn.Conv2d(256, 256, kernel_size=3,
                                          padding=1, bias=False)
        self.bottleneck_bn2 = nn.BatchNorm2d(256)

        # 1x1 expand
        self.bottleneck_conv3 = nn.Conv2d(256, 512, kernel_size=1,
                                          bias=False)
        self.bottleneck_bn3 = nn.BatchNorm2d(512)

        # Adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Improved classifier with deeper features
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 2048),  # After adaptive pooling: 512 channels
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
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

        # NEW: Conv Layer 6: 256 -> 384 channels
        x = F.relu(self.bn6(self.conv6(x)))  # 3x3 -> 3x3

        # NEW: Conv Layer 7: 384 -> 384 channels (WITH SKIP CONNECTION)
        identity = x  # Same dimensions, direct skip
        x = self.bn7(self.conv7(x))
        x = x + identity  # Skip connection
        x = F.relu(x)

        # NEW: Conv Layer 8: 384 -> 512 channels
        x = F.relu(self.bn8(self.conv8(x)))  # 3x3 -> 3x3
        x = self.maxpool4(x)                 # 3x3 -> 1x1

        # NEW: Conv Layer 9: 512 -> 512 channels (WITH SKIP CONNECTION)
        identity = x  # Same dimensions, direct skip
        x = self.bn9(self.conv9(x))
        x = x + identity  # Skip connection
        x = F.relu(x)

        # NEW: Conv Layer 10: 512 -> 512 channels (WITH SKIP CONNECTION)
        identity = x  # Same dimensions, direct skip
        x = self.bn10(self.conv10(x))
        x = x + identity  # Skip connection
        x = F.relu(x)

        # BOTTLENECK BLOCK (512 -> 256 -> 512) WITH SKIP CONNECTION
        identity = x  # Save input for skip connection

        # 1x1 convolution to reduce channels: 512 -> 256
        x = F.relu(self.bottleneck_bn1(self.bottleneck_conv1(x)))

        # 3x3 convolution to process features
        x = F.relu(self.bottleneck_bn2(self.bottleneck_conv2(x)))

        # 1x1 convolution to expand channels: 256 -> 512
        x = self.bottleneck_bn3(self.bottleneck_conv3(x))

        # Add skip connection
        x = x + identity
        x = F.relu(x)

        # Adaptive pooling to ensure 1x1 spatial size (replaces maxpool5)
        x = self.adaptive_pool(x)  # -> 512x1x1

        # Classification
        x = torch.flatten(x, 1)  # -> 512
        x = self.classifier(x)
        return x
