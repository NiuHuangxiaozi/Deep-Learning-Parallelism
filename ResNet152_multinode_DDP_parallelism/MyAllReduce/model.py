#
# Implementation of AlexNet for illustrative purposes. The train.py driver
# can import AlexNet from here or directly from torchvision.
#
# Taken from torchvision.models.alexnet:
# https://pytorch.org/docs/1.6.0/_modules/torchvision/models/alexnet.html#alexnet


import torch
import torch.nn as nn
import torchvision.models as models



#ever used train on cifar10
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



#train on flower102
class Resnet_large(nn.Module):
    def __init__(self, model):
        super(Resnet_large, self).__init__()
        self.resnet = nn.Sequential(*list(model.children())[:-1])
        # 可以选择冻结卷积层
        # for p in self.parameters():
        #     p.requires_grad = False
        self.conv1=nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=2,padding=0)
        self.fc = nn.Linear(in_features=2048, out_features=102)

    def forward(self, x):
        x=self.conv1(x)
        x = self.resnet(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x