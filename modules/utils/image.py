import torch

from PIL import Image
from torchvision import transforms


class ImageUtils(object):
    """
    The utility class for image processing.
    """

    @staticmethod
    def load_image(filename: str, size: int) -> torch.FloatTensor:
        """
        Load and resize image.

        :param filename: (str) The image file name.
        :param size: (int) The desired size. The smaller edge of the image will be matched to this number.
        :return: (torch.FloatTensor) The image tensor.
        """
        # Load image and convert RBG to BGR due to the pre-trained VGG limitation.
        # TODO: it may be better to convert color space in VGG module becuase Face Net uses RGB color space.
        image = Image.open(filename).convert('RGB')
        r, g, b = image.split()
        image = Image.merge(image.mode, (b, g, r))
        # Resize and transform to Torch.
        load_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = load_transform(image)
        image = image.unsqueeze(0)
        return image

    @staticmethod
    def save_image(image: torch.FloatTensor, filename: str):
        """
        Save image from torch tensor.

        :param filename: (str) The output image file name.
        :param image: (torch.FloatTensor) The image tensor to output.
        """
        # Transform to pillow image.
        image = image.squeeze(0)
        save_transform = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            transforms.ToPILImage()
        ])
        image = save_transform(image)
        # Cnvert BGR to RGB and save image.
        # TODO: change implementation according to the above.
        b, g, r = image.split()
        image = Image.merge(image.mode, (r, g, b))
        image.save(filename)
