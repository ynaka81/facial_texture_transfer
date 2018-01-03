import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models


class Vgg16(nn.Module):

    __MAPPING = {
        1: '1_1', 3: '1_2', 6: '1_3',
        8: '2_1', 11: '2_2', 13: '2_3',
        15: '3_1', 18: '3_2', 20: '3_3', 22: '3_4',
        25: '4_1', 27: '4_2', 29: '4_3'
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
