import argparse
import os

from modules.transfer.facial_transfer import FacialTransfer


def make_args():
    parser = argparse.ArgumentParser(description='Transfer style image to content image.')
    parser.add_argument('content', type=str, help='The content image path.')
    parser.add_argument('style', type=str, help='The style image path.')
    parser.add_argument('--content-weight', type=float, default=5, help='The weight of content loss.')
    parser.add_argument('--style-weight', type=float, default=100, help='The weight of style loss.')
    parser.add_argument('--tv-weight', type=float, default=1e-3, help='The weight of tv loss.')
    parser.add_argument('--use-gpu', '-g', action='store_true', help='Use GPU to transfer.')
    parser.add_argument('--content-size', '-c', type=int, default=256, help='The desired size of content image.')
    parser.add_argument('--style-size', '-s', type=int, default=256, help='The desired size of style image.')
    parser.add_argument('--iterations', '-i', type=int, default=1000, help='The number of iterations.')
    parser.add_argument('--log-interval', type=int, default=100, help='The interval of log output.')
    parser.add_argument('--output-interval', type=int, default=10, help='The interval of output image.')
    args = parser.parse_args()
    return args


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
