import os

from option import make_args
from modules.transfer.facial_transfer import FacialTransfer
from modules.losses.mrf_texture_style_loss import MRFTextureStyleLoss


def main():
    """
    The main function.
    """
    # Get arguments.
    args = make_args()
    # Make output directory.
    output = 'output/mrf'
    os.makedirs(output, exist_ok=True)
    # Transfer style to content.
    transfer = FacialTransfer(style_loss_class=MRFTextureStyleLoss, gpu=args.use_gpu, output=output)
    transfer.run(args.content, args.style,
                 content_weight=1e-4, style_weight=1e-2, tv_weight=1e2, **vars(args))


if __name__ == '__main__':
    main()
