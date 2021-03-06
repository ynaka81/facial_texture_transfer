import math

import torch
from torch.autograd import Variable
from torch.nn import Upsample

from modules.models.inception_resnet_v1 import create_net


class FaceIdentificationLoss(object):
    """
    L2 distance of the face feature.

    :param content_image: (torch.Tensor) The content image.
    :param gpu: (bool) Whether or not to use GPU.
    """

    def __init__(self, content_image: torch.FloatTensor, gpu: bool):
        # Load Face Net.
        self.__face_net = create_net()
        self.__face_net.load_state_dict(torch.load('face_net.pth'))
        if gpu:
            self.__face_net.cuda()
        self.__face_net.eval()
        # Calculate content feature.
        self.__upsample = Upsample(scale_factor=2, mode='bilinear')
        self.__std_numel = Variable(torch.Tensor([1 / math.sqrt(content_image.numel())]), requires_grad=False)
        if gpu:
            self.__std_numel = self.__std_numel.cuda()
            self.__upsample.cuda()
        self.__f_x_c = Variable(self.__embedding(content_image).data, requires_grad=False)
        if gpu:
            self.__f_x_c = self.__f_x_c.cuda()

    def __embedding(self, image):
        # Upsample when the image is too small.
        b, ch, h, w = image.size()
        if min(h, w) < 80:
            image = self.__upsample(image)
        # Prewhiten image.
        mean = image.mean()
        std = image.std()
        std_adj = torch.max(std, self.__std_numel)
        image = (image - mean) / std_adj
        # Return Face Net embeddings.
        return self.__face_net(image)

    def __call__(self, image):
        """
        Calculate L2 distances.

        :param image: (torch.Tensor) The output image.
        """
        f_x_t = self.__embedding(image)
        return 1 - f_x_t.dot(self.__f_x_c)
