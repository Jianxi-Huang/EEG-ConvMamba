import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, num_classes, chans, samples=1000, dropout_rate=0.5, kernel_length=64, F1=8,
                 F2=16,):
        super(EEGNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1, kernel_size=(chans, 1), groups=F1, bias=False),  # groups=F1 for depthWiseConv
            nn.BatchNorm2d(F1),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
            # for SeparableCon2D
            # SeparableConv2D(F1, F2, kernel1_size=(1, 16), bias=False),
            nn.Conv2d(F1, F1, kernel_size=(1, 16), groups=F1, bias=False),  # groups=F1 for depthWiseConv
            nn.BatchNorm2d(F1),
            nn.ELU(inplace=True),
            nn.Conv2d(F1, F2, kernel_size=(1, 1), groups=1, bias=False),  # point-wise cnn
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=dropout_rate),
        )
        out = torch.ones((1, 1, chans, samples))
        out = self.features(out)
        n_out_time = out.cpu().data.numpy().shape
        self.classifier = nn.Linear(n_out_time[-1] * n_out_time[-2] * n_out_time[-3], num_classes)

    def forward(self, x):
        conv_features = self.features(x)
        features = torch.flatten(conv_features, 1)
        cls = self.classifier(features)
        return cls