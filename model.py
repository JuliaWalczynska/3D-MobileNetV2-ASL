'''
3D-MobileNetV2 model.
Code based on Torchvision repository, Efficient-3DCNNs repository and Kopuklu et el., 2019 paper.
Torchvision: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
Efficient-3DCNNs: https://github.com/okankop/Efficient-3DCNNs/blob/master/models/mobilenetv2.py
Kopuklu, O., Kose, N., Gunduz, A., & Rigoll, G. (2019).
Resource efficient 3d convolutional neural networks. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision Workshops (pp. 0-0).
'''
import math
import torch.nn as nn
import torch.nn.functional as F


def convBNReLu(inp, oup, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), groups=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)

        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(convBNReLu(inp, hidden_dim, 1, stride=(1, 1, 1), padding=(0, 0, 0)))

        layers.extend([
            # dw
            convBNReLu(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm3d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=100, width_mult=3.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, (1, 1, 1)],
            [6, 24, 2, (2, 2, 2)],
            [6, 32, 3, (2, 2, 2)],
            [6, 64, 4, (2, 2, 2)],
            [6, 96, 3, (1, 1, 1)],
            [6, 160, 3, (2, 2, 2)],
            [6, 320, 1, (1, 1, 1)],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [convBNReLu(3, input_channel, stride=(1, 2, 2))]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(convBNReLu(input_channel, self.last_channel, 1, 1, 0))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

