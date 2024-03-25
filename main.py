"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from munch import Munch
from torch.backends import cudnn
import torch

from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.solver import Solver
from core.load_args import load_args
from core.eval import eval_xai


def subdirs(dname):
    """
    Return a list of subdirectories in the given directory.

    Parameters:
    - dname (str): The directory path.

    Returns:
    - list: List of subdirectories in the given directory.
    """
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    """
    Main function to execute StarGAN v2 training or evaluation.

    Parameters:
    - args (argparse.Namespace): Command-line arguments.

    Returns:
    - None
    """
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    # Create a Solver instance
    solver = Solver(args)

    if args.mode == 'train':
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.val_img_dir)) == args.num_domains

        # Get training data loader
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             img_channels=args.img_channels,
                                             data_range_norm=args.data_range_norm))

        # Get test data loader for evaluation during training
        test_loaders = Munch(src=get_test_loader(root=args.src_dir,
                                                 img_size=args.img_size,
                                                 batch_size=args.val_batch_size,
                                                 shuffle=False,
                                                 num_workers=args.num_workers,
                                                 img_channels=args.img_channels,
                                                 data_range_norm=args.data_range_norm))

        # Start training
        solver.train(loaders, test_loaders)
        print('Training done.')

    elif args.mode == 'eval':
        # Evaluate explainability using XAI methods
        eval_xai(args, use_true_labels=True, experiment_type='global_beta')
        eval_xai(args, use_true_labels=True, experiment_type='faithfulness')
        print('Evaluation done.')

    else:
        raise NotImplementedError

    exit()


if __name__ == '__main__':
    # Load command-line arguments
    args = load_args()
    # Execute the main function
    main(args)
