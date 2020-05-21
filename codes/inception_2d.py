import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InceptionBlock(nn.Module): #X, f, filters, stage, block):
    def __init__(self, in_channels, out_channels, epsilon = 0.001):
        super(InceptionBlock, self).__init__()

        def pad_same(f):
            pad = np.ceil((f - 1) / 2)
            return (int(pad),int(pad)) #TODO check same for even filter and with s

        # conv 1x1 - is like "avarage" of previouses layers - preserves spatial dimensions, reduces depth
        self.branch1_conv1x1 = nn.Conv2d(in_channels=in_channels, kernel_size=1, out_channels= 96, stride=1)
        self.branch1_bn_1 = nn.BatchNorm2d(96, eps=epsilon)
        self.branch1_conv3x3 = nn.Conv2d(in_channels=96, kernel_size=3, out_channels=128, stride=1, padding=pad_same(3))
        self.branch1_bn_2 = nn.BatchNorm2d(128, eps=epsilon)

        self.branch2_conv1x1 = nn.Conv2d(in_channels=in_channels, kernel_size=1, out_channels=16, stride=1)
        self.branch2_bn_1 = nn.BatchNorm2d(16, eps=epsilon)
        self.branch2_conv5x5 = nn.Conv2d(in_channels=16, kernel_size=5, out_channels=32, stride=1, padding=pad_same(5))
        self.branch2_bn_2 = nn.BatchNorm2d(32, eps=epsilon)

        self.branch3_maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=pad_same(3))
        self.branch3_conv1x1 = nn.Conv2d(in_channels=in_channels, kernel_size=1, out_channels=32, stride=1)
        self.branch3_bn_1 = nn.BatchNorm2d(32, eps=epsilon)

        self.branch4_conv1x1 = nn.Conv2d(in_channels=in_channels, kernel_size=1, out_channels= 64, stride=1)
        self.branch4_bn_1 = nn.BatchNorm2d(64, eps=epsilon)


    def forward(self, x):
        #branch 1:
        branch1 = self.branch1_conv1x1(x)
        branch1 = self.branch1_bn_1(branch1)
        branch1 = F.relu(branch1)

        branch1 = self.branch1_conv3x3(branch1)
        branch1 = self.branch1_bn_2(branch1)
        branch1 = F.relu(branch1)

        # branch 2:
        branch2 = self.branch2_conv1x1(x)
        branch2 = self.branch2_bn_1(branch2)
        branch2 = F.relu(branch2)

        branch2 = self.branch2_conv5x5(branch2)
        branch2 = self.branch2_bn_2(branch2)
        branch2 = F.relu(branch2)

        # branch 3:
        branch3 = self.branch3_maxpool(x)
        branch3 = self.branch3_conv1x1(branch3)
        branch3 = self.branch3_bn_1(branch3)
        branch3 = F.relu(branch3)


        # branch 4:
        branch4 = self.branch4_conv1x1(x)
        branch4 = self.branch4_bn_1(branch4)
        branch4 = F.relu(branch4)


        # out_channels = 256
        return torch.cat([branch1, branch2, branch3, branch4], 1)



####################################################################



class InceptionsNet(nn.Module):
    def __init__(self, num_classes = 2, input_size = 256):
        super(InceptionsNet, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.conv7x7_1 = nn.Conv2d(in_channels=3, kernel_size=7, out_channels=32, stride=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv5x5_1 = nn.Conv2d(in_channels=32, kernel_size=5, out_channels=64, stride=1)
        self.bn2 = nn.BatchNorm2d(64)

        # create layers of inception blocks
        self.unit1 = InceptionBlock(in_channels=64, out_channels=256)
        self.unit2 = InceptionBlock(in_channels=256, out_channels=256)
        self.unit3 = InceptionBlock(in_channels=256, out_channels=256)

        self.fc1 = nn.Linear(in_features=30976, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=num_classes)


    def forward(self,x):
        x = self.conv7x7_1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        x = self.conv5x5_1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)

        x = self.unit1(x) #is adding more channels in each block-so need do bootleneck layers / maxpoolong
        x = self.unit2(x)
        x = self.unit3(x)
        x = F.avg_pool2d(x, kernel_size=7, stride=2, padding=0)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x



