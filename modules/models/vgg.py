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
        self.__vgg = models.vgg16(pretrained=True).features

    def forward(self, X):
        h = X
        features = {}
        for no, layer in enumerate(self.__vgg):
            h = layer(h)
            if no in self.__MAPPING:
                features[self.__MAPPING[no]] = h
        return features
