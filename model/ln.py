import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class ResNetBackbone1D(nn.Module):
    def __init__(self, layers, input_channels=1):
        super(ResNetBackbone1D, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv1d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(16, layers[0])
        self.layer2 = self._make_layer(32, layers[1], stride=2)
        self.layer3 = self._make_layer(64, layers[2], stride=2)
        self.layer4 = self._make_layer(128, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

        layers = [ResidualBlock1D(self.in_channels, out_channels, stride=stride, downsample=downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class LatentSpaceNetwork(nn.Module):
    def __init__(self, num_lstm_layers=1, hidden_size=128, latent_size=64, resnet_layers=[2, 2, 2, 2]):
        super(LatentSpaceNetwork, self).__init__()
        self.resnet = ResNetBackbone1D(layers=resnet_layers, input_channels=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        batch_size, sequence_length = x.shape
        x = x.view(batch_size, 1, sequence_length)
        x = self.resnet(x)
        x, _ = self.lstm(x.unsqueeze(1))  
        latent = self.fc(x[:, -1, :])  # chx index
        return latent