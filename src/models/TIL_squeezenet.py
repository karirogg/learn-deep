import torch
import torch.nn as nn
import torch.nn.init as init


class Fire(nn.Module):
    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int,
    ) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1
        )
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x)),
            ],
            1,
        )


class Task_IL_SqueezeNet(nn.Module):
    def __init__(
        self, num_classes_per_task: int = 50, num_tasks: int = 2, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.num_classes = num_classes_per_task
        self.num_tasks = num_tasks

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        # Task-specific classifiers, one per task
        self.task_classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Conv2d(512, self.num_classes, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                for _ in range(num_tasks)
            ]
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                for classifier in self.task_classifiers:
                    final_conv = classifier[1]
                    init.normal_(final_conv.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        assert 0 <= task_id < self.num_tasks, f"Invalid task_id: {task_id}"
        x = self.features(x)
        x = self.task_classifiers[task_id](x)
        return torch.flatten(x, 1)
