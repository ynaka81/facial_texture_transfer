import torch
import torch.nn as nn
import torch.nn.functional as F


def create_net():
    return InceptionResnetV1()


class GlobalAvgPool2d(nn.Module):

    def __init__(self, count_include_pad=False):
        super(GlobalAvgPool2d, self).__init__()
        self.count_include_pad = count_include_pad

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = F.avg_pool2d(x, (h, w), count_include_pad=self.count_include_pad)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.005,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicLinear(nn.Module):

    def __init__(self, in_dimension, out_dimension):
        super(BasicLinear, self).__init__()
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        self.linear = nn.Linear(in_dimension, out_dimension, bias=False)
        self.bn = nn.BatchNorm1d(out_dimension,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.005,  # default pytorch value
                                 affine=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResnetV1(nn.Module):

    def __init__(self,
                 drop_rate=0.2,
                 bottleneck_layer_size=128):

        # original image size is (299, 299) but we can use any sufficiently large image size
        # like 160 thanks to GlobalAvgPool2d

        super(InceptionResnetV1, self).__init__()

        self.drop_rate = drop_rate
        self.bottleneck_layer_size = bottleneck_layer_size

        # 149 x 149 x 32
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)

        # 147 x 147 x 32
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)

        # 147 x 147 x 64
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # 73 x 73 x 64
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)

        # 73 x 73 x 80
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)

        # 71 x 71 x 192
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)

        # 35 x 35 x 256
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)

        # 5 x Inception-resnet-A
        # 35 x 35 x 256
        self.mixed_5a = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17))

        # Reduction-A
        # 17 x 17 x (384 + 256 + 256) = 17 x 17 x 896
        self.mixed_6a = Mixed_6a()

        # 10 x Inception-Resnet-B
        # 17 x 17 x 896
        self.mixed_6b = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )

        # Reduction-B
        # 8 x 8 x 1792
        self.mixed_7a = Mixed_7a()

        # 5 x Inception-Resnet-C
        # 8 x 8 x 1792
        self.mixed_8a = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20))

        # Mixed 8b
        # 8 x 8 x 1792
        self.mixed_8b = Block8(scale=1.0, noReLU=True)

        # Logits
        # 1 x 1 x 1792
        self.avgpool_1a = GlobalAvgPool2d(count_include_pad=False)

        # 1792
        self.dropout_1a = nn.Dropout2d(drop_rate)

        # bottleneck_layer_size
        self.fc_1a = BasicLinear(1792, bottleneck_layer_size)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.mixed_5a(x)
        x = self.mixed_6a(x)
        x = self.mixed_6b(x)
        x = self.mixed_7a(x)
        x = self.mixed_8a(x)
        x = self.mixed_8b(x)
        x = self.avgpool_1a(x)
        x = x.view(-1, 1792)
        x = self.dropout_1a(x)
        x = self.fc_1a(x)

        x = x / x.norm(dim=1).unsqueeze(1).expand_as(x)  # mapping to unit ball

        return x
