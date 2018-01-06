import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models


class Vgg16(nn.Module):

    __MAPPING = {
        1: '1_1', 3: '1_2',
        6: '2_1', 8: '2_2',
        11: '3_1', 13: '3_2', 15: '3_3',
        18: '4_1', 20: '4_2', 22: '4_3',
        25: '5_1', 27: '5_2', 29: '5_3'
    }

    def __init__(self):
        super(Vgg16, self).__init__()
        self.__mean = Variable(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
        self.__std = Variable(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)
        self.__vgg = models.vgg16(pretrained=True).features

    def cuda(self) -> nn.Module:
        """
        Move all model parameters and buffers to the GPU.

        :return: (nn.Module) Return myself.
        """
        super().cuda()
        self.__vgg.cuda()
        self.__mean = self.__mean.cuda()
        self.__std = self.__std.cuda()
        return self

    def forward(self, X):
        h = (X - self.__mean) / self.__std
        features = {}
        for no, layer in enumerate(self.__vgg):
            h = layer(h)
            if no in self.__MAPPING:
                features[self.__MAPPING[no]] = h
        return features
