#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings

# ================= 新增：1D 通道注意力机制模块 =================
class SEBlock1D(nn.Module):
    """
    一维通道注意力机制 (Squeeze-and-Excitation)
    作用：过滤工况相关噪声，提纯故障特征，提升域泛化能力
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        # 将自适应权重乘回原特征，压制无用通道，放大核心通道
        return x * y.expand_as(x)
# ===============================================================

# ----------------------------inputsize >=28-------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(CNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=15),  # 16, 26 ,26
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )  # 32, 12,12     (24-2) /2 +1

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),  # 64,10,10
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),  # 128,8,8
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4))  # 128, 4,4
            
        # 【创新点植入】：在这里实例化注意力机制，对接 layer4 的 128 个通道
        self.attention = SEBlock1D(channel=128, reduction=16)

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 【核心拦截】：在特征展平前，用注意力机制净化 128 维特征！
        x = self.attention(x)
        
        x = x.view(x.size(0), -1)
        x = self.layer5(x)

        return x

# convnet without the last layer
class cnn_features(nn.Module):
    def __init__(self, pretrained=False):
        super(cnn_features, self).__init__()
        self.model_cnn = CNN(pretrained)
        self.__in_features = 256

    def forward(self, x):
        x = self.model_cnn(x)
        return x

    def output_num(self):
        return self.__in_features