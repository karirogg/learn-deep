import torch
from torch import nn

# from src.metrics.mc_dropout import mc_dropout_inference

"""
Dataset Transformations

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(MEAN, STD)])
test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


Training settings

batch_size = 64
epochs = 100
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
"""

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    squeeze_net = torch.hub.load(
        "pytorch/vision:v0.10.0", "squeezenet1_1", pretrained=False
    )
    num_classes = 10
    model = squeeze_net
    model.features[0] = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1
    )  # Adjust first layer for 32x32 images
    model.features[2] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    )
    model = model.to(device)
    PATH = "src\models\cifar\experiments\squeezenet100.pth"
    model.load_state_dict(torch.load(PATH, weights_only=True, map_location=device))
