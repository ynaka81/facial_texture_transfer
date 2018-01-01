import argparse


def make_args():
    parser = argparse.ArgumentParser(description='Transfer style image to content image.')
    parser.add_argument('content', type=str, help='The content image path.')
    parser.add_argument('style', type=str, help='The style image path.')
    parser.add_argument('--use-gpu', '-g', action='store_true', help='Use GPU to transfer.')
    parser.add_argument('--content-size', '-c', type=int, default=256, help='The desired size of content image.')
    parser.add_argument('--style-size', '-s', type=int, default=256, help='The desired size of style image.')
    parser.add_argument('--iterations', '-i', type=int, default=1000, help='The number of iterations.')
    parser.add_argument('--log-interval', type=int, default=100, help='The interval of log output.')
    parser.add_argument('--output-interval', type=int, default=10, help='The interval of output image.')
    args = parser.parse_args()
    return args
