import torch
import torch.nn as nn


class TinyNerfModel(nn.Module):
    def __init__(self, filter_size: int = 128, num_encoding_functions: int = 6) -> None:
        super().__init__()

        self.fc_layer1 = nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
        self.fc_layer2 = nn.Linear(filter_size, filter_size)
        self.fc_layer3 = nn.Linear(filter_size, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc_layer1(x))
        x = torch.relu(self.fc_layer2(x))
        x = self.fc_layer3(x)

        return x
