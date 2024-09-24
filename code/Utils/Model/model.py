import torch
import argparse
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    filename='model.log',
    level=logging.INFO)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--read', action='store_true')
    parser.add_argument('--extract', action='store_true')

    args = parser.parse_args()
    return args


def main(args):

    model_path = '/home1/hxl/Documents/OneLLM/weights/consolidated.00-of-01.pth'

    model_weights = torch.load(model_path)

    if args.read:

        for key, val in model_weights.items():
            print(key)

    if args.extract:

        prex = ''
        output_path = './LLaMA'

if __name__ == '__main__':
    args = parse_args()
    main(args)