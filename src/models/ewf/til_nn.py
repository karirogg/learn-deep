import torch
import torch.nn as nn
import torch.nn.init as init



class TaskILNN(nn.Module):
    def __init__(
        self, num_tasks: int = 4, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks

        self.features = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64)
        )

        # Task-specific classifiers, one per task
        self.task_classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(64, 1)
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

        return torch.flatten(x, 1).reshape(-1)
