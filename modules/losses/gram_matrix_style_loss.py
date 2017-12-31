import typing

import torch
from torch.autograd import Variable
from torch.nn import MSELoss


class GramMatrixStyleLoss(object):
    """
    L2 distance between gram matrix of content features and that of output features.

    :param style_features: (dict) The feature maps of style image.
    :param gpu: (bool) Whether or not to use GPU.
    """

    STYLE_FEATURES = ('1_1', '2_1', '3_1', '4_1')

    def __init__(self, style_features: typing.Dict[str, torch.FloatTensor], gpu: bool):
        self.__g_x_s = {}
        for feature_name in self.STYLE_FEATURES:
            gram_matrix = self.__gram_matrix(style_features[feature_name])
            self.__g_x_s[feature_name] = Variable(gram_matrix.data, requires_grad=False)
        self.__mse_loss = MSELoss()
        if gpu:
            for g_x_s in self.__g_x_s.values():
                g_x_s.cuda()
            self.__mse_loss.cuda()

    def __gram_matrix(self, feature):
        b, ch, h, w = feature.size()
        feature = feature.view(b, ch, w * h)
        feature_t = feature.transpose(1, 2)
        gram = feature.bmm(feature_t) / (ch * h * w)
        return gram

    def __call__(self, features):
        """
        Calculate L2 distances.

        :param features: (dict) The features maps of output image.
        """
        loss = 0
        for feature_name in self.STYLE_FEATURES:
            gram_matrix = self.__gram_matrix(features[feature_name])
            loss += self.__mse_loss(gram_matrix, self.__g_x_s[feature_name])
        return loss
