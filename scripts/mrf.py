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
    # Set parameters if it is not specified by command.
    if args.iterations is None:
        args.iterations = 200
    if args.content_weight is None:
        args.content_weight = 1e-4
    if args.face_weight is None:
        args.face_weight = 1
    if args.style_weight is None:
        args.style_weight = 1e-2
    if args.tv_weight is None:
        args.tv_weight = 1e2
    # Make output directory.
    output = 'output/mrf'
    os.makedirs(output, exist_ok=True)
    # Transfer style to content.
    transfer = FacialTransfer(style_loss_class=MRFTextureStyleLoss, gpu=args.use_gpu, output=output)
    transfer.run(args.content, args.style, **vars(args))


if __name__ == '__main__':
    main()
