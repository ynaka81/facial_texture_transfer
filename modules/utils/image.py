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
        # Load image.
        image = Image.open(filename).convert('RGB')
        # Resize and transform to Torch.
        load_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
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
        # Clamp image value.
        image.clamp_(0, 1)
        # Transform to pillow image.
        image = image.squeeze(0)
        save_transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        image = save_transform(image)
        # Save image.
        image.save(filename)
