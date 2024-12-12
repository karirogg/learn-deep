from torch import nn
import torch.nn.functional as F

class CIFAR_CNN(nn.Module):
    def __init__(self, cfg):
        super(CIFAR_CNN, self).__init__()
        
        # extract hyperparams
        num_classes = cfg["num_classes"]

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 32x32 -> 32x32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 32x32 -> 32x32
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 16x16 -> 16x16
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # Flatten after 3 pools (4x4 feature maps)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout to reduce overfitting
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = F.relu(self.conv1(x))  # 32x32 -> 32x32
        x = self.pool(F.relu(self.conv2(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv3(x)))  # 16x16 -> 8x8
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 256 * 4 * 4)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
