import pickle
from tqdm import tqdm
from core.load_args import load_args
from core.xai_utils import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = load_args()


def eval_xai(args, use_true_labels=True, experiment_type='global_beta'):
    
    assert experiment_type in ('global_beta', 'faithfulness')
    assert os.path.isdir(args.checkpoint_dir)
    data_name = args.data_name
    mission_name = args.mission_name
    classifier_type = args.classifier_type
    batch_size = 1

    args.resume_iter = get_last_resume_iter(args.checkpoint_dir)
    threshold2plot = 30

    if experiment_type in 'global_beta':
        global_beta = True
    else:
        global_beta = False
    
    one_heatmap_image = False
    show_color = False  # True
    show_only_attr = True  # False #

    save_class_agnostic = True  # False #

    lrp_classifier_type = 'classifier3' if 'discriminator' in classifier_type else classifier_type

    methods_list = ['rand', 'dxai', 'LayerGradCam', 'GuidedGradCam',
                    'GradientShap', 'lrp_relu', 'InternalInfluence', 'IntegratedGradients', 'Lime']
    beta_list = [0, 0.01, 0.05, 0.1, 0.15, 0.2]
    if global_beta:
        beta_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    beta_list.sort()

    _, test_set = make_datasets(data_name, args.img_channels, args.img_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    classes = np.asarray(test_set.classes)
    num_of_classes = len(classes)
    args.num_domains = num_of_classes
    args.max_eval_iter = min(args.max_eval_iter, int(len(test_set)/batch_size))
    iters_num = min(args.max_eval_iter*batch_size, len(test_set))
    run_name = './xai_output/xai_'+data_name+'_'+mission_name
    branch_path = args.checkpoint_dir+os.sep+format(args.resume_iter, '06d')+'_nets_ema.ckpt'
    details_dict_path = run_name + os.sep + str(args.resume_iter)+'_details_dict'+'_'+classifier_type+'_'+str(iters_num)+'_iters'+'.pkl'

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
    print('number of test examples:     ', len(test_set))
    print('number of current examples:  ', min(args.max_eval_iter*batch_size, len(test_set)))
    print('PATH is:     ', branch_path)
    print('run name is: ', run_name)
    print('details-dict path is: ', details_dict_path.replace(run_name, ''))
    print(args.resume_iter)

    nets = load_branch_gan_networks(args, branch_path, device)
    resnet_classifier = load_classifier('resnet18', data_name, args, num_of_classes, device)
    classifier = load_classifier(classifier_type, data_name, args, num_of_classes, device, nets)
    model = nets.discriminator.module if 'discriminator' in classifier_type else classifier
    if not os.path.isdir(run_name):
        os.makedirs(run_name)

    attributions2show = []

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
            y_true, y_pred, entropy_list = [], [], []
            torch.manual_seed(123)
            np.random.seed(123)
            print('')
            print('testing... , beta = ', beta)
            images2show = []
            attributions2show_per_method = []
            probs_hists = torch.zeros(num_of_classes, num_of_classes).to(device)
            class_counter = torch.zeros(num_of_classes).to(device)
            
            with torch.no_grad():
                for ii, data in enumerate(tqdm(test_loader)):
                    if ii >= 1.01*args.max_eval_iter and args.max_eval_iter <= threshold2plot or ii >= args.max_eval_iter > threshold2plot or \
                            args.max_eval_iter <= threshold2plot < len(images2show)*batch_size:
                        break
                    images, labels = data[0].to(device), data[1].to(device)
                    y_true.extend(labels.data.cpu().numpy())
                    labels2xai = make_xai_labels(labels, use_true_labels, num_of_classes)

                    attributions, mask, x2class = make_attribution_map_and_mask(xai_method, method, images, labels2xai,
                                                    args, nets, beta, run_name, ii, classifier_type, 
                                                    save_class_agnostic=save_class_agnostic, global_beta=global_beta)

                    outputs = nets.discriminator(x2class) if 'discriminator' in classifier_type else classifier(x2class)

                    _, predicted = torch.max(outputs, dim=1)
                    y_pred.extend(predicted.data.cpu().numpy())
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    probs = F.softmax(outputs, dim=1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(1).mean()
                    entropy_list.append(entropy)
                    probs_hists[labels] += probs
                    class_counter[labels] += labels.size(0)

                    if xai_method not in 'rand' and aa == 1 and len(images2show)*batch_size <= threshold2plot and args.max_eval_iter <= threshold2plot:
                        images2show, attributions2show_per_method, attributions2show = save_heatmaps(images, attributions,
                                      images2show, attributions2show_per_method, attributions2show, args, args.img_channels,
                                      run_name, xai_method, batch_size, args.max_eval_iter, labels2xai, labels, ii, classifier_type, one_heatmap_image, show_only_attr=show_only_attr, show_color=show_color)

            details_dict[xai_method].probs_hists[details_dict[xai_method].betas.index(beta)] = np.round(1e3 * (probs_hists / class_counter).detach().cpu().numpy()) / 1e3
            details_dict[xai_method].values[details_dict[xai_method].betas.index(beta)] = np.round(1e3 * correct / total) / 1e3
            print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))
            #print_entropy_detatils(y_true, y_pred, entropy_list)

            with open(details_dict_path, "wb") as fp:
                pickle.dump(details_dict, fp)

            print('')
            print(data_name, '-', xai_method)
            with open(details_dict_path, "wb") as fp:
                pickle.dump(details_dict, fp)
        compute_AUC(details_dict, run_name, args, classifier_type, iters_num, global_beta, T=max(beta_list))

    save_accuracy_figure(details_dict, args, num_of_classes, data_name, mission_name, run_name, iters_num, classifier_type, global_beta)

    '''
    for key in details_dict.keys():# 'dxai'
        #print([h for h in details_dict[key].probs_hists])
        probs_hists_stack = np.stack(details_dict[key].probs_hists, axis=0).reshape((-1, num_of_classes))
        print(probs_hists_stack.shape)
        print(probs_hists_stack)
        
        plt.rcParams.update({'font.size': 16})

        f,a = plt.subplots(len(details_dict[key].probs_hists), num_of_classes)
        f.set_size_inches((5*num_of_classes, 3*len(details_dict[key].probs_hists)), forward=False)
        a = a.ravel()
        
        for idx,ax in enumerate(a):
            ax.bar(classes, probs_hists_stack[idx], width=0.5)
            if idx < num_of_classes:
                #ax.set_title("Probabilities of '"+classes[idx%num_of_classes]+"' Label", fontsize=15)
                ax.set_title(classes[0] + '   '+classes[1] + '   '+classes[2], fontsize=18)
            if idx % num_of_classes == 0:
                ax.set_ylabel(r'$\beta$=' + str(details_dict[key].betas[idx // num_of_classes]), rotation='horizontal',  fontsize=20, labelpad=55)
            ax.set_ylim(bottom=0, top=1)
            ax.set_yticks([0, 1])
            ax.set_xticks([])
            plt.subplots_adjust(wspace=1.5) 
        #plt.tight_layout()
        plt.savefig(run_name + os.sep + str(args.resume_iter)+'_'+key+'_histograms_'+classifier_type+'_'+str(iters_num)+'_iters'+'.png', format='png', dpi=500)#, bbox_inches="tight")
        #plt.show()
        plt.close()
    '''
    print('Done')
    return 0
