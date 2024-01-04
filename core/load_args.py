import argparse


def load_args():
    parser = argparse.ArgumentParser()
    # changed
    base_directory = '../Data/'
    base_sample_directory = 'assets/'
    parser.add_argument('--train_img_dir', type=str, default=base_directory + 'train', help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default=base_directory + 'val', help='Directory containing validation images')
    parser.add_argument('--lambda_reg', type=float, default=1, help='Weight for R1 regularization')
    parser.add_argument('--lambda_adv', type=float, default=2, help='adversarial loss for generator')
    parser.add_argument('--lambda_class_fake', type=float, default=2, help='adversarial loss for generator')

    parser.add_argument('--lambda_rec', type=float, default=4, help='Weight for reconstruction loss')
    parser.add_argument('--lambda_grad_rec', type=float, default=0, help='Weight for gradients similarity loss')
    parser.add_argument('--lambda_dis_rec', type=float, default=4, help='Weight for dis-rec loss')

    parser.add_argument('--lambda_orth', type=float, default=1, help='Weight for orthogonal regularization')

    parser.add_argument('--softmax_temp', type=float, default=1, help='Softmax temperature of classification losses')

    parser.add_argument('--zero_st', type=int, default=1, help='If 1, set distinction map to zero when style transfer had made')
    parser.add_argument('--data_range_norm', type=int, default=1, help='If 1, set range of images value from -1 to 1, else from 0 to 1' )
    parser.add_argument('--out_features', type=int, default=1, help='')

    parser.add_argument('--use_pretrained_classifier', type=int, default=1, help='if use pre-trained classifier for xai')
    parser.add_argument('--classifier_type', type=str, default='', help='stargan or resnet18 or vgg16 or classifier2')
    parser.add_argument('--classifier_ckpt_path', type=str, default='', help='checkpoints path to classifier weights')

    parser.add_argument('--style_dim', type=int, default=64, help='Style code dimension')
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--img_channels', type=int, default=3, help='Image channels, can be 3 or 1')
    parser.add_argument('--alpha_blend', type=int, default=1, help='If 1, use alpha blend process')
    # new
    parser.add_argument('--num_branches', type=int, default=6)
    parser.add_argument('--branch_dimin', type=int, default=7)  # generator's dimin equals num_branches * branch_dimin (63=9*7 by default)
    parser.add_argument('--data_name', type=str, default='')
    parser.add_argument('--mission_name', type=str, default='')

    parser.add_argument('--style_per_block', type=int, default=True, help='indicates how and whether style code (of a specific image) is should be mixed upon testing')

    # original stargan-v2
    parser.add_argument('--mode', default='train', help='train or eval')
    parser.add_argument('--num_AdaInResBlks', type=int, default=1, help='number of AdaIn blocks, which will be assigned as the number of style vectors needed for inference')

    # model arguments
    parser.add_argument('--num_domains', type=int, default=2, help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension')

    parser.add_argument('--w_hpf', type=float, default=0, help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--total_iters', type=int, default=800000, help='Number of total iterations')
   
    parser.add_argument('--max_eval_iter', type=int, default=1500, help='Number of total iterations')
        
    parser.add_argument('--resume_iter', type=int, default=0, help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=30,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for mapping-style network')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')

    # directory for testing

    parser.add_argument('--src_dir', type=str, default=base_sample_directory, help='Directory containing input source images')

    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=3000)  # 10000
    parser.add_argument('--save_every', type=int, default=10000)  # 10000 50000

    args = parser.parse_args()
    if args.num_branches == 1:
        args.alpha_blend = False
    return args
