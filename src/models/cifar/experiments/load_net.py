import torch
from torch import nn


def load_pretrained_squeezenet(
    cifar10=False, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
):

    if cifar10:
        num_classes = 10
        PATH = "src\models\cifar\experiments\mod_sq_net_cifar10_50epochs.pth"
    else:
        num_classes = 100
        PATH = "src\models\cifar\experiments\mod_sq_net_cifar100_50epochs.pth"

    model = torch.hub.load("pytorch/vision:v0.10.0", "squeezenet1_1", pretrained=False)
    model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    model.features[2] = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
    model.features[5] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    model.features[8] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model = model.to(device)

    model.load_state_dict(torch.load(PATH, weights_only=True, map_location=device))

    return model
