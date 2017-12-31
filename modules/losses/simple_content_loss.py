import typing

import torch
from torch.autograd import Variable
from torch.nn import MSELoss


class SimpleContentLoss(object):
    """
    Simple L2 distance of the feature maps at each convolution layers.

    :param content_features: (dict) The feature maps of content image.
    :param gpu: (bool) Whether or not to use GPU.
    """

    CONTENT_FEATURE = '4_2'

    def __init__(self, content_features: typing.Dict[str, torch.FloatTensor], gpu: bool):
        content_feature = content_features[self.CONTENT_FEATURE]
        self.__f_x_c = Variable(content_feature.data, requires_grad=False)
        self.__mse_loss = MSELoss()
        if gpu:
            self.__f_x_c = self.__f_x_c.cuda()
            self.__mse_loss.cuda()

    def __call__(self, features):
        """
        Calculate L2 distances.

        :param features: (dict) The features maps of output image.
        """
        f_x_t = features[self.CONTENT_FEATURE]
        return self.__mse_loss(f_x_t, self.__f_x_c)
