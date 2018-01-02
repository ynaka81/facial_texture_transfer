import typing

import torch
from torch.autograd import Variable
from torch.nn import functional as F


class MRFTextureStyleLoss(object):
    """
    L2 distance between a local feature patch of output image and the most similar patch in style image.

    :param style_features: (dict) The feature maps of style image.
    :param gpu: (bool) Whether or not to use GPU.
    """

    STYLE_FEATURES = ('3_1', '4_1')
    K = 3

    def __init__(self, style_features: typing.Dict[str, torch.FloatTensor], gpu: bool):
        self.__p_x_s = {}
        self.__pn_x_s = {}
        self.__w_p = {}
        for feature_name in self.STYLE_FEATURES:
            feature = style_features[feature_name]
            _, ch, _, _ = feature.size()
            w_p = torch.eye(ch * self.K * self.K).view(-1, ch, self.K, self.K)
            self.__w_p[ch] = Variable(w_p, requires_grad=False)
            p_x_s = self.__patch(feature)
            self.__p_x_s[feature_name] = p_x_s
            self.__pn_x_s[feature_name] = self.__normalize_patch(p_x_s)
        if gpu:
            for x in (self.__p_x_s, self.__pn_x_s, self.__w_p):
                for k, v in x.items():
                    x[k] = v.cuda()

    def __patch(self, feature):
        _, ch, _, _ = feature.size()
        return F.conv2d(feature, self.__w_p[ch]).view(ch, -1)

    def __normalize_patch(self, patch):
        return patch / patch.norm(p=2, dim=0)

    def __nearest_neighbor(self, p_x_t, p_x_s, pn_x_s):
        ch, N = p_x_t.size()
        pn_x_t = self.__normalize_patch(p_x_t)
        correlation = pn_x_t.t().mm(pn_x_s)
        _, NN = correlation.max(dim=1)
        return p_x_s[:, NN]

    def __call__(self, features):
        """
        Calculate L2 distances.

        :param features: (dict) The features maps of output image.
        """
        loss = 0
        for feature_name in self.STYLE_FEATURES:
            p_x_t = self.__patch(features[feature_name])
            p_nn_x_s = self.__nearest_neighbor(p_x_t, self.__p_x_s[feature_name], self.__pn_x_s[feature_name])
            loss += torch.dist(p_x_t, p_nn_x_s) / p_nn_x_s.numel()
        return loss
