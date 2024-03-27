import pickle
from tqdm import tqdm
from core.load_args import load_args
from core.xai_utils import *
from core.data_loader import get_test_loader
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = load_args()


def eval_xai(args, use_true_labels=True, experiment_type='global_beta', save_images=True):
    """
    Evaluate explainability methods on a trained model.

    Args:
        args (argparse.Namespace): Command-line arguments.
        use_true_labels (bool, optional): Whether to use true labels for XAI methods. Default is True.
        experiment_type (str, optional): Type of XAI experiment, either 'global_beta' or 'faithfulness'. Default is 'global_beta'.
        use_true_labels (bool, optional): Whether to save images of heatmaps, class agnostic and distinction parts in the output folder. Default is True.

    Returns:
        int: Return value indicating the completion status (0 for success).

    Raises:
        AssertionError: If experiment_type is not one of ('global_beta', 'faithfulness') or if checkpoint_dir does not exist.

    Note:
        This function assumes the presence of various helper functions for data loading, network loading,
        and XAI method evaluation.

    """

    assert experiment_type in ('global_beta', 'faithfulness')
    assert os.path.isdir(args.checkpoint_dir)

    # Set various parameters and paths
    data_name = args.data_name
    running_name = args.running_name
    classifier_type = args.classifier_type
    batch_size = 1

    # Get the last checkpoint iteration
    args.resume_iter = get_last_resume_iter(args.checkpoint_dir)
    threshold2plot = 30

    # Check experiment type
    if experiment_type in 'global_beta':
        global_beta = True
    else:
        global_beta = False
    
    # Set LRP classifier type
    lrp_classifier_type = 'classifier3' if 'discriminator' in classifier_type else classifier_type

    # Define XAI methods and beta values
    methods_list = ['rand', 'dxai', 'LayerGradCam', 'GuidedGradCam',
                    'GradientShap', 'lrp_relu', 'InternalInfluence', 'IntegratedGradients', 'Lime']
                    
    beta_list = list(np.round(100*np.arange(0, 0.25, 0.05))/100)
    if global_beta:
        beta_list = list(np.round(100*np.arange (0, 1.1, 0.1))/100)
    beta_list.sort()

    # Load test dataset
   
    len_test_set = sum([len(files) for r, d, files in os.walk(args.val_img_dir)])
    classes = sorted(os.listdir(args.val_img_dir))
    test_loader = get_test_loader(root=args.val_img_dir,
                     img_size=args.img_size,
                     batch_size=batch_size,
                     shuffle=True,
                     num_workers=args.num_workers,
                     img_channels=args.img_channels,
                     data_range_norm=args.data_range_norm)
                     
    # Get class information
    num_of_classes = args.num_domains
    args.max_eval_iter = min(args.max_eval_iter, int(len_test_set/batch_size))
    iters_num = min(args.max_eval_iter*batch_size, len_test_set)
    out_folder = './xai_output/xai_'+data_name+'_'+running_name
    branch_path = args.checkpoint_dir + os.sep + format(args.resume_iter, '06d') + '_nets_ema.ckpt'
    details_dict_path = out_folder + os.sep + str(args.resume_iter) + '_details_dict' + '_' + classifier_type + '_' + str(iters_num) + '_iters' + '.pkl'

    if global_beta:
        details_dict_path = details_dict_path.replace('details_dict', 'global_beta_details_dict')

    if os.path.isfile(details_dict_path):
        with open(details_dict_path, "rb") as fp:
            details_dict = pickle.load(fp)
    else:
        details_dict = {}

    print_dictionary(details_dict)

    print('data name:                   ', data_name)
    print('classes:                     ', classes)
    print('number of classes:           ', num_of_classes)
    print('number of test examples:     ', len_test_set)
    print('number of current examples:  ', min(args.max_eval_iter*batch_size, len_test_set))
    print('PATH is:     ', branch_path)
    print('run name is: ', out_folder)
    print('details-dict path is: ', details_dict_path.replace(out_folder, ''))
    print(args.resume_iter)

    # Load GAN networks and classifiers
    nets = load_branch_gan_networks(args, branch_path, device)
    classifier = load_classifier(classifier_type, data_name, args, num_of_classes, device, nets)
    model = nets.discriminator.module if 'discriminator' in classifier_type else classifier
    
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    for xx, xai_method in enumerate(methods_list):
        if 'Lime' in xai_method and batch_size*args.max_eval_iter > threshold2plot and len(methods_list) > 1:
            continue
        
        print('###########################')
        print('mask mode: ', xai_method)
        print('')
        if xai_method not in details_dict.keys():
            details_dict = boot_method(details_dict, xai_method, global_beta, num_of_classes=num_of_classes)
        
        method = prepare_method_of_xai(xai_method, model, classifier_type, lrp_classifier_type, data_name, args, num_of_classes, device)
        for aa, beta in enumerate(beta_list):
            
            if beta not in details_dict[xai_method].betas:
                details_dict = update_beta_and_value(details_dict, xai_method, beta)
            
            if details_dict[xai_method].values[details_dict[xai_method].betas.index(beta)] is not None and args.max_eval_iter > threshold2plot:
                continue
                      
            correct, total = 0, 0
            y_true, y_pred = [], []
            torch.manual_seed(123)
            np.random.seed(123)
            print('')
            print('testing... , beta = ', beta)
            
            probs_hists = torch.zeros(num_of_classes, num_of_classes).to(device)
            class_counter = torch.zeros(num_of_classes).to(device)
            
            with torch.no_grad():
                for ii, data in enumerate(tqdm(test_loader)):
                    if ii >= args.max_eval_iter and args.max_eval_iter <= threshold2plot or ii >= args.max_eval_iter > threshold2plot:
                        break
                    images, labels = data[0].to(device), data[1].to(device)
                    y_true.extend(labels.data.cpu().numpy())
                    labels2xai = make_xai_labels(labels, use_true_labels, num_of_classes)

                    attributions, mask, x2class = make_attribution_map_and_mask(xai_method, method, images, labels2xai,
                                                    args, nets, beta, out_folder, ii, classifier_type, 
                                                    save_images=save_images, global_beta=global_beta)

                    outputs = nets.discriminator(x2class) if 'discriminator' in classifier_type else classifier(x2class)

                    _, predicted = torch.max(outputs, dim=1)
                    y_pred.extend(predicted.data.cpu().numpy())
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    probs = F.softmax(outputs, dim=1)
                    probs_hists[labels] += probs
                    class_counter[labels] += labels.size(0)

                    
            details_dict[xai_method].probs_hists[details_dict[xai_method].betas.index(beta)] = np.round(1e3 * (probs_hists / class_counter).detach().cpu().numpy()) / 1e3
            details_dict[xai_method].values[details_dict[xai_method].betas.index(beta)] = np.round(1e3 * correct / total) / 1e3
            print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))

            with open(details_dict_path, "wb") as fp:
                pickle.dump(details_dict, fp)

            print('')
            print(data_name, '-', xai_method)
            with open(details_dict_path, "wb") as fp:
                pickle.dump(details_dict, fp)
        compute_AUC(details_dict, out_folder, args, classifier_type, iters_num, global_beta, T=max(beta_list))

    save_accuracy_figure(details_dict, args, num_of_classes, data_name, running_name, out_folder, iters_num, classifier_type, global_beta)

    print('Done')
    return 0
    
