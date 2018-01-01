import os

from tqdm import tqdm, trange
from torch.autograd import Variable
from torch.optim import LBFGS

from modules.losses.simple_content_loss import SimpleContentLoss
from modules.losses.gram_matrix_style_loss import GramMatrixStyleLoss
from modules.losses.total_variation_regularization import TotalVariationRegularization
from modules.models.vgg import Vgg16
from modules.utils.image import ImageUtils


class FacialTransfer(object):
    """
    The implementation of "Photo-realistic Facial Texture Transfer".

    :param gpu: (bool) Whether or not to use GPU.
    :param output: (str) The output directory name.
    """

    def __init__(self, gpu: bool, output: str):
        self.gpu = gpu
        self.output = output
        # Load pre-train VGG model.
        self.vgg = Vgg16()
        if gpu:
            self.vgg.cuda()

    def run(self, content_image: str, style_image: str,
            content_weight: float, style_weight: float, tv_weight: float,
            content_size: int, style_size: int, iterations: int,
            log_interval: int, output_interval: int, **kwargs):
        """
        Start transfer.

        :param content_image: (str) The content image filename.
        :param style_image: (str) The style image filename.
        :param content_weight: (float) The weight of content loss.
        :param style_weight: (float) The weight of style loss.
        :param style_weight: (float) The weight of tv loss.
        :param content_size: (int) The desired size of content image.
        :param style_size: (int) The desired size of style image.
        :param iterations: (int) The number of iterations.
        :param log_interval: (int) The interval of log output.
        :param output_interval: (int) The interval of output image.
        """
        # Load images.
        content_image = ImageUtils.load_image(content_image, content_size)
        content_image = Variable(content_image, requires_grad=False)
        style_image = ImageUtils.load_image(style_image, style_size)
        style_image = Variable(style_image, requires_grad=False)
        # Convert to GPU mode.
        if self.gpu:
            content_image = content_image.cuda()
            style_image = style_image.cuda()
        # Initialize optimizer.
        target_image = Variable(content_image.data, requires_grad=True)
        optimizer = LBFGS([target_image], lr=1, history_size=10)
        # Setup losses.
        content_features = self.vgg(content_image)
        content_loss = SimpleContentLoss(content_features, self.gpu)
        style_features = self.vgg(style_image)
        style_loss = GramMatrixStyleLoss(style_features, self.gpu)
        tv_loss = TotalVariationRegularization()
        # Optimize the image.
        for i in trange(iterations):

            def closure():
                ImageUtils.clamp_image(target_image)
                # Initialize gradation.
                optimizer.zero_grad()
                # Calculate losses.
                output_features = self.vgg(target_image)
                total_loss = content_weight * content_loss(output_features) + style_weight * style_loss(output_features) + tv_weight * tv_loss(target_image)
                total_loss.backward()
                return total_loss

            # Optimize.
            loss = optimizer.step(closure)
            if i % log_interval == 0:
                tqdm.write('loss = ' + str(loss.data.cpu().numpy()[0]))
            # Output optimizing image.
            if i % output_interval == 0:
                output_image = target_image.clone()
                if self.gpu:
                    output_image = target_image.cpu()
                ImageUtils.save_image(output_image.data, os.path.join(self.output, f'{i}.png'))
