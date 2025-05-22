# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class Connect4Net(nn.Module):
    def __init__(self, input_channels=2, board_height=6, board_width=7, num_blocks=3):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_height * board_width, board_width)

        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_height * board_width, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, state_tensor):
        # state_tensor: [2, 6, 7] numpy array => [1, 2, 6, 7] tensor
        self.eval()
        with torch.no_grad():
            x = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0)
            p, v = self.forward(x)
            return torch.exp(p[0]).numpy(), v.item()

