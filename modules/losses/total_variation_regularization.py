import torch


class TotalVariationRegularization(object):
    """
    Total variation regularization of output image.
    """

    def __call__(self, image):
        """
        Calculate total variation.

        :param features: (dict) The features maps of output image.
        """
        loss_w = torch.sum((image[:, :, :, :-1] - image[:, :, :, 1:]) ** 2)
        loss_h = torch.sum((image[:, :, :-1, :] - image[:, :, 1:, :]) ** 2)
        return loss_w + loss_h
