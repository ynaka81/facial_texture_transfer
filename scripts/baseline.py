import os

from option import make_args
from modules.transfer.facial_transfer import FacialTransfer
from modules.losses.simple_content_loss import SimpleContentLoss


def main():
    """
    The main function.
    """
    # Get arguments.
    args = make_args()
    # Make output directory.
    output = 'output/baseline'
    os.makedirs(output, exist_ok=True)
    # Transfer style to content.
    transfer = FacialTransfer(content_loss_class=SimpleContentLoss, gpu=args.use_gpu, output=output)
    transfer.run(args.content, args.style,
                 content_weight=1e-3, style_weight=1, tv_weight=10, **vars(args))


if __name__ == '__main__':
    main()
