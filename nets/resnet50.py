#resnet50L利用conv block（残差边上有卷积和标准化，可以通过改变卷积步长和通道数从而改变输出维数）
# 和identity block（无卷积，输出输入维度一样，进行串联，加深网络）堆叠起来
#faster rcnn主干特征提取网络得到一个共享特征层shape:38*38*1024（输入图片600*600*3）（分成38*38网格，每个网格9个鲜艳框）

import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

class Bottleneck(nn.Module):   #瓶颈结构，更好的提取特征，加深网络，可以减少参数量
    expansion = 4#残差中卷积核个数有没有变化 64——256
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        #卷积，卷积大小1，压缩通道数
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #卷积2，卷积大小3，特征提取
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #卷积3，卷积大小1，拓张通道数
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x#捷径输出值
        #1*1卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #3*3卷积，提取特征
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #1*1卷积，扩张通道数
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:#如果残差边上有卷积，则卷积
            residual = self.downsample(x)

        out += residual#加上捷径再激活
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        #-----------------------------------#
        #   假设输入进来的图片是600,600,3
        #-----------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64
        #对输入进来的图片进行了一个大小，步长2，通道64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)#标准化
        self.relu = nn.ReLU(inplace=True)#激活函数

        # 300,300,64 -> 150,150,64
        #最大池化，步长2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256#多个Block->多个layer->网络（blackbone）
        #基准通道数64

        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None#下采样为none
        #-------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        #-------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(  #残差边，下采样
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []#定义一个空列表
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):#将网络结构压入
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):#前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    #传入了一个数组，对应了resnet50的结构
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    #----------------------------------------------------------------------------#
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    #----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    #----第五次压缩的内容和池化模型分给分类器，进行池化后，进行分类和回归预测------------------------------------------------------------------------#
    classifier = list([model.layer4, model.avgpool])
    
    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier
