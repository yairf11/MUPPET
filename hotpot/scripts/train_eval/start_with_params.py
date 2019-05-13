import argparse

from hotpot.trainer import start_training_with_params
from hotpot.model_dir import ModelDir


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('name', help='name of output to examine')
    parser.add_argument('--eval', "-e", action="store_true")
    args = parser.parse_args()

    start_training_with_params(ModelDir(args.name), start_eval=args.eval)


if __name__ == "__main__":
    # import os
    #
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
