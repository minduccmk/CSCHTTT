import torch.nn as nn


class MLP(nn.Module):
    """2 hidden layer (100 unit) + 1 lớp FC đầu ra.

    `features` = các lớp đầu (kiến thức chung, dùng cho fine-tuning trong FedKT)
    `classifier` = lớp FC cuối (bị loại khỏi bước copy của FedKT)
    """

    def __init__(self, in_dim: int, hidden: int = 100, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x))
