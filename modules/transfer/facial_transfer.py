import os

from tqdm import tqdm, trange
from torch.autograd import Variable
from torch.optim import LBFGS

from modules.losses.simple_content_loss import SimpleContentLoss
from modules.losses.face_identification_loss import FaceIdentificationLoss
from modules.losses.total_variation_regularization import TotalVariationRegularization
from modules.models.vgg import Vgg16
from modules.utils.image import ImageUtils


class FacialTransfer(object):
    """
    The implementation of "Photo-realistic Facial Texture Transfer".

    :param style_loss_class: (object) The style loss class.
    :param gpu: (bool) Whether or not to use GPU.
    :param output: (str) The output directory name.
    """

    def __init__(self, style_loss_class: object, gpu: bool, output: str):
        self.style_loss_class = style_loss_class
        self.gpu = gpu
        self.output = output
        # Load pre-train VGG model.
        self.vgg = Vgg16()
        if gpu:
            self.vgg.cuda()

    def run(self, content_image: str, style_image: str,
            content_weight: float, face_weight: float, style_weight: float, tv_weight: float,
            content_size: int, style_size: int, iterations: int,
            log_interval: int, output_interval: int, **kwargs):
        """
        Start transfer.

        :param content_image: (str) The content image filename.
        :param style_image: (str) The style image filename.
        :param content_weight: (float) The weight of content loss.
        :param face_weight: (float) The weight of face loss.
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
        # Multi-resolution optimization process.
        base = 0
        target_image_r = None
        for stride in tqdm([4, 2, 1]):
            if content_size // stride < 64:
                continue
            # Setup images.
            content_image_r = content_image[:, :, ::stride, ::stride]
            style_image_r = style_image[:, :, ::stride, ::stride]
            if target_image_r is None:  # Initial stage.
                target_image_r = Variable(content_image_r.data, requires_grad=True)
            else:
                b, ch, h, w = target_image_r.size()
                target_image_r_data = target_image_r.data
                target_image_r_data = target_image_r_data.unsqueeze(3).repeat(1, 1, 1, 2, 1).view(b, ch, h * 2, w)
                target_image_r_data = target_image_r_data.unsqueeze(4).repeat(1, 1, 1, 1, 2).view(b, ch, h * 2, w * 2)
                target_image_r = Variable(target_image_r_data, requires_grad=True)
            # Initialize optimizer.
            optimizer = LBFGS([target_image_r], lr=1, max_iter=10)
            # Setup losses.
            content_features = self.vgg(content_image_r)
            content_loss = SimpleContentLoss(content_features, self.gpu)
            face_loss = FaceIdentificationLoss(content_image_r, self.gpu)
            style_features = self.vgg(style_image_r)
            style_loss = self.style_loss_class(style_features, self.gpu)
            tv_loss = TotalVariationRegularization()
            # Optimize the image.
            for i in trange(base, iterations + base):
                self.call_count = 0

                def closure():
                    ImageUtils.clamp_image(target_image_r.data)
                    # Initialize gradation.
                    optimizer.zero_grad()
                    # Calculate losses.
                    output_features = self.vgg(target_image_r)
                    content_loss_i = content_loss(output_features)
                    face_loss_i = face_loss(target_image_r)
                    style_loss_i = style_loss(output_features)
                    tv_loss_i = tv_loss(target_image_r)
                    total_loss = content_weight * content_loss_i + face_weight * face_loss_i + style_weight * style_loss_i + tv_weight * tv_loss_i
                    total_loss.backward(retain_graph=True)
                    # Log each loss.
                    if i % log_interval == 0 and self.call_count == 0:
                        tqdm.write('content_loss = ' + str(content_loss_i.data.cpu().numpy()[0]) + ', ' +
                                   'face_loss = ' + str(face_loss_i.data.cpu().numpy()[0]) + ', ' +
                                   'style_loss = ' + str(style_loss_i.data.cpu().numpy()[0]) + ', ' +
                                   'tv_loss = ' + str(tv_loss_i.data.cpu().numpy()[0]) + ', ' +
                                   'total_loss = ' + str(total_loss.data.cpu().numpy()[0]))
                    self.call_count += 1
                    return total_loss

                # Optimize.
                optimizer.step(closure)
                # Output optimizing image.
                if (i + 1) % output_interval == 0:
                    output_image = target_image_r.clone()
                    if self.gpu:
                        output_image = target_image_r.cpu()
                    ImageUtils.save_image(output_image.data, os.path.join(self.output, f'{i + 1}.png'))
            base += iterations
