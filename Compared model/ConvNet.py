import torch
import torch.nn as nn


def square_activation(x):
    return torch.square(x)


def safe_log(x):
    return torch.clip(torch.log(x), min=1e-7, max=1e7)

class ConvNet(nn.Module):
    def __init__(self, num_classes, chans, samples=1000):
        super(ShallowConvNet, self).__init__()
        self.conv_nums = 40
        self.features = nn.Sequential(
            nn.Conv2d(1, self.conv_nums, (1, 25)),
            nn.Conv2d(self.conv_nums, self.conv_nums, (chans, 1), bias=False),
            nn.BatchNorm2d(self.conv_nums)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout()
        out = torch.ones((1, 1, chans, samples))
        out = self.features(out)
        out = self.avgpool(out)
        n_out_time = out.cpu().data.numpy().shape
        self.classifier = nn.Linear(n_out_time[-1] * n_out_time[-2] * n_out_time[-3], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = square_activation(x)
        x = self.avgpool(x)
        x = safe_log(x)
        x = self.dropout(x)
        features = torch.flatten(x, 1)
        cls = self.classifier(features)
        return cls

