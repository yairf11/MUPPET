import argparse

from hotpot.trainer import resume_training
from hotpot.model_dir import ModelDir


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('name', help='name of output to examine')
    parser.add_argument('--eval', "-e", action="store_true")
    parser.add_argument('--n-async', "-n", type=int, default=8)
    parser.add_argument('--dev-b', type=int, default=None)
    args = parser.parse_args()

    resume_training(ModelDir(args.name), start_eval=args.eval, async_encoding=args.n_async, dev_batch_size=args.dev_b)


if __name__ == "__main__":
    # import os
    #
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()