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
        loss_w = torch.dist(image[:, :, :, :-1], image[:, :, :, 1:])
        loss_h = torch.dist(image[:, :, :-1, :], image[:, :, 1:, :])
        return (loss_w + loss_h) / image.numel()
