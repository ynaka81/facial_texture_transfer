import os

from option import make_args
from modules.transfer.facial_transfer import FacialTransfer


def main():
    """
    The main function.
    """
    # Get arguments.
    args = make_args()
    # Make output directory.
    output = 'output'
    os.makedirs(output, exist_ok=True)
    # Transfer style to content.
    transfer = FacialTransfer(gpu=args.use_gpu, output=output)
    transfer.run(args.content, args.style, **vars(args))


if __name__ == '__main__':
    main()
