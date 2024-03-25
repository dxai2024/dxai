import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import os
import random
from core.branch_utils import sum_groups

from core.model import build_model, load_pretrained_classifier
import matplotlib.pyplot as plt
from core.utils import save_image, make_anomaly_heatmap, tensor_contrast_stretch, tensor2ndarray255
from torch.nn import functional as F
from skimage.segmentation import slic, mark_boundaries
from captum._utils.models.linear_model import SkLearnLasso

from captum.attr import (
    GradientShap, DeepLift,
    IntegratedGradients,
    Saliency, LRP, GuidedGradCam,
    LayerGradCam, InternalInfluence, Lime,
)


class method_values:
  def __init__(self, betas=[], values=[], confusion_matrix=[], num_of_classes=2):
    self.betas = []
    self.values = []
    # self.confusion_matrix = confusion_matrix
    self.SNR_dB = []
    self.std = []
    self.att_accuracy = np.nan
    self.attributions = []
    self.probs_hists = []#torch.zeros(num_of_classes, num_of_classes)
    self.class_counter = []#torch.zeros(num_of_classes)


def make_datasets(data_name, img_channels=1, img_size=256):
    train_data_folder = '../Data' + os.sep + data_name + os.sep + 'train'
    test_data_folder = '../Data' + os.sep + data_name + os.sep + 'val'

    class MyRotateTransform:
        def __init__(self):
            self.angles = [0, 90, -90, 180]

        def __call__(self, x):
            angle = random.choice(self.angles)
            return TF.rotate(x, angle)

    train_transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        MyRotateTransform(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * img_channels,
                             std=[0.5] * img_channels),
    ])
    test_transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * img_channels,
                             std=[0.5] * img_channels),
    ])
    if img_channels == 1:
        train_transform.transforms.insert(0, transforms.Grayscale(num_output_channels=img_channels))
        test_transform.transforms.insert(0, transforms.Grayscale(num_output_channels=img_channels))
    trainset = torchvision.datasets.ImageFolder(root=train_data_folder, transform=train_transform)
    testset = torchvision.datasets.ImageFolder(root=test_data_folder, transform=test_transform)
    return trainset, testset


def load_branch_gan_networks(args, branch_path, device):
    nets, nets_ema = build_model(args)
    branch_ckp = torch.load(branch_path)
    nets_ema.generator.load_state_dict(branch_ckp['generator'])
    nets_ema.discriminator.load_state_dict(branch_ckp['discriminator'])
    nets_ema.mapping_network.load_state_dict(branch_ckp['mapping_network'])
    nets_ema.generator.to(device)
    nets_ema.mapping_network.to(device)
    nets_ema.discriminator.to(device)
    return nets_ema


def prepare_method_of_xai(xai_method, model, classifier_type, lrp_classifier_type, data_name, args, num_of_classes, device):

    if 'resnet' in classifier_type:
        layer = model.layer3[-1].conv2
    elif 'discriminator' in classifier_type:
        repeat_num = int(np.log2(args.img_size)) - 2
        layer = model.main[repeat_num-1].conv2  # model.main[xai_layer]
    elif 'simple_classifier' in classifier_type:
        layer = model.main[-2] # model.main[xai_layer]
    if 'lrp' in xai_method:
        LRP_classifier = load_classifier(lrp_classifier_type, data_name, args, num_of_classes, device)
        method = LRP(model=LRP_classifier)
    elif 'branch_saliency' in xai_method:
        method = Saliency(model)
    elif 'Saliency' in xai_method:
        method = Saliency(model)
    elif 'IntegratedGradients' in xai_method:
        method = IntegratedGradients(model)
    elif 'GuidedGradCam' in xai_method:
        method = GuidedGradCam(model, layer=layer)
    elif 'InternalInfluence' in xai_method:
        if 'resnet' in classifier_type:
            layer = model.layer4[0].conv1
        elif 'discriminator' in classifier_type:
            repeat_num = int(np.log2(args.img_size)) - 2
            layer = model.main[repeat_num-3].conv1  # model.main[xai_layer]
        elif 'simple_classifier' in classifier_type:
            layer = model.main[-2] # model.main[xai_layer]
        method = InternalInfluence(forward_func=model.forward, layer=layer)
    elif 'LayerGradCam' in xai_method:
        method = LayerGradCam(model, layer=layer)
    elif 'GradientShap' in xai_method:
        method = GradientShap(model.forward)
    elif 'Lime' in xai_method:
        method = Lime(model.forward, interpretable_model=SkLearnLasso(alpha=0.01))
    elif 'DeepLift' in xai_method:
        method = DeepLift(model)
    else:
        method = None
    return method


def make_attribution_map_and_mask(xai_method, method, images, labels2xai, args, nets, beta, out_folder, index, classifier_type, save_images=False, agnostic_mode=False, global_beta=False):
    masks = None
    if xai_method in 'dxai':
        z_trg = torch.randn(images.size(0), args.latent_dim).to(images.device)
        s_trg = nets.mapping_network(z_trg, labels2xai)
        x_fake, Psi, Phi, Res, Masks = nets.generator(images, s_trg)
        attributions = Psi[:, 0:args.img_channels]#.abs()
        class_agnostic = sum_groups(Psi[:, args.img_channels::], args.img_channels)

    elif 'branch_saliency' in xai_method:
        z_trg = torch.randn(images.size(0), args.latent_dim).to(images.device)
        s_trg = nets.mapping_network(z_trg, labels2xai)
        x_fake, Psi, Phi, Res, Masks = nets.generator(images, s_trg)
        branch_output = Psi[:, 0:args.img_channels]#.abs()
        attributions = method.attribute(branch_output, target=labels2xai)
    elif 'Saliency' in xai_method:
        attributions = method.attribute(images, target=labels2xai)
    elif 'LayerGradCam' in xai_method:
        attributions = method.attribute(images, target=labels2xai, relu_attributions=True)
        attributions = F.interpolate(attributions, size=(args.img_size, args.img_size),
                                     mode='bilinear', align_corners=True)
    elif 'IntegratedGradients' in xai_method:
        attributions = method.attribute(images, target=labels2xai)
    elif 'GradientShap' in xai_method:
        attributions = method.attribute(images, target=labels2xai, baselines=torch.zeros_like(images), n_samples=25)
    elif 'Lime' in xai_method:
        features = slic2tensor(images, n_segments=64)
        attributions = method.attribute(images, target=labels2xai, baselines=torch.zeros_like(images), n_samples=1000, feature_mask=features)
    elif 'InternalInfluence' in xai_method:
        attributions = F.relu(method.attribute(images, target=labels2xai))
        attributions = F.interpolate(attributions.mean(dim=1, keepdim=True),
                                     size=(args.img_size, args.img_size), mode='bilinear')
    elif 'GuidedGradCam' in xai_method:
        attributions = method.attribute(images, target=labels2xai, interpolate_mode='bilinear')
    elif 'lrp' in xai_method:
        attributions = method.attribute(images, target=labels2xai)
    elif 'DeepLift' in xai_method:
        attribution = method.attribute(images, target=labels2xai)
    if 'rand' in xai_method:
        if beta <= 1:
            masks = torch.bernoulli(beta*torch.ones(images.size(0), 1, images.size(2), images.size(3))).to(images.device).detach()
        attributions = torch.rand(images.size(0), 1, images.size(2), images.size(3)).to(images.device)
    elif 'blank' in xai_method:
        if beta <= 1:
            masks = torch.bernoulli(beta*torch.ones(images.size(0), 1, images.size(2), images.size(3))).to(images.device).detach()
        attributions = torch.ones_like(images)
    else:
        if 'relu' in xai_method:
            attributions = F.relu(attributions)
        if agnostic_mode:
            if xai_method in 'dxai':
                attributions = class_agnostic
            else:
                attributions = tensor_contrast_strech(F.relu(attributions))
                attributions = 1 - attributions
        if beta <= 1:
            masks = torch.zeros(images.size(0), 1, images.size(2), images.size(3)).to(images.device)
            if beta < 1:
                intence = attributions.abs().mean(dim=1, keepdim=True)
                values, _ = torch.sort(intence.view(attributions.size(0), -1), descending=True)
                masks[attributions.abs().mean(dim=1, keepdim=True) > values[:, int(beta*values.size(1))].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)] = 1
            
    if global_beta:
        if 'dxai' not in xai_method: 
            att_image = attributions / (attributions.abs().max()+1e-8) * images
            attribution_normed = att_image
        else:
            attribution_normed = attributions

        CD_global = (beta*attribution_normed) 
        
        x2class = images - CD_global

    else:
        x2class = (1-masks)*images 
    
    if save_images and index < 3e2:
        distinction_agnostic_folder = out_folder + os.sep + str(args.resume_iter) + '_distinction_agnostic_'+classifier_type+ os.sep + xai_method
        if global_beta:
            distinction_agnostic_folder = distinction_agnostic_folder + '_beta_'+str(beta).replace('.', '_')
        if not os.path.isdir(distinction_agnostic_folder):
            os.makedirs(distinction_agnostic_folder)
        if 'dxai' not in xai_method:
            attribution_strech = F.relu(attributions)/ (F.relu(attributions).max() + 1e-8)
            CA_mask = 1 - attribution_strech
            CA = CA_mask * images
            CD = attribution_strech * images
        else:
            attribution_strech = attributions
            CA = class_agnostic
            CD = attributions
        if global_beta:
            save_image(torch.cat((images, CD_global, x2class), dim=-1), 3,
                       distinction_agnostic_folder + os.sep + str(labels2xai.item())+'_'+str(index+1) + '.png')
        else:
            if images.size(1)==1:
                save_image(torch.cat((images.repeat_interleave(repeats=3, dim=1), CD.repeat_interleave(repeats=3, dim=1), make_anomaly_heatmap(attribution_strech, images)), dim=-1), 3,
                           distinction_agnostic_folder + os.sep + str(labels2xai.item())+'_'+str(index+1) + '_heatmap.png')
            else:
                save_image(torch.cat((images, CD, make_anomaly_heatmap(attribution_strech, images)), dim=-1), 3,
                           distinction_agnostic_folder + os.sep + str(labels2xai.item())+'_'+str(index+1) + '_heatmap.png')
            save_image(torch.cat((images, CD, CA), dim=-1), 3,
                       distinction_agnostic_folder + os.sep + str(labels2xai.item())+'_'+str(index+1) + '.png')
    return attributions, masks, x2class.float()


def make_xai_labels(labels, use_true_labels, num_of_classes):
    labels2xai = labels.clone()
    if not use_true_labels:
        for jj in range(len(labels)):
            class_list = list(range(0, num_of_classes))
            class_list.remove(int(labels[jj].detach().cpu().numpy()))
            labels2xai[jj] = random.choice(class_list)
    return labels2xai
    
    
def save_accuracy_figure(details_dict, args, num_of_classes, data_name, running_name, out_folder, iters_num, classifier_type, global_beta=False):
    print('save figure...')
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    plt.figure()
    for key in details_dict.keys():
        if details_dict[key].values.count(None) < len(details_dict[key].values) - 1:
            plt.plot(details_dict[key].betas, details_dict[key].values, label=key)
    plt.grid(True)
    plt.title(data_name + ' - ' + running_name)
  
    plt.axhline(y=1 / num_of_classes, color='r', linestyle='--')
    plt.ylabel('accuracy', fontsize=17)
    plt.xlabel('beta', fontsize=17)
    plt.legend(fontsize=9)
    plt.ylim((0, 1))
    if global_beta:
        experiment_name = 'global_beta'
    else:
        experiment_name = 'faithness'
    plt.savefig(out_folder + os.sep + str(args.resume_iter)+'_'+experiment_name+'_'+classifier_type+'_'+str(iters_num)+'_iters'+'.png', format='png', bbox_inches="tight")
    plt.close()
    print('figure saved..')


def save_heatmaps(images, attributions, images2show, attributions2show_per_method, attributions2show, args, img_channels,
                  out_folder, xai_method, batch_size, max_iter, labels2xai, labels, index, classifier_type, one_heatmap_image=False, show_only_attr=True, show_color=True, agnostic_mode=False):
    images2show.append(images)
    attributions = attributions / (attributions.abs().max()+1e-8) if xai_method not in 'dxai' and (attributions.abs().max()<0.9 or attributions.abs().max()>1) else attributions
    attributions2show_per_method.append(attributions)# / attributions.abs().max())
    if len(attributions2show_per_method) * batch_size >= 1 and one_heatmap_image or index == max_iter - 1 and not one_heatmap_image:
        if len(attributions2show) == 0:
            if img_channels < 3 and not show_only_attr:
                attributions2show += [torch.cat(images2show, dim=0).repeat_interleave(repeats=3, dim=1)]
            else:
                attributions2show += [torch.cat(images2show, dim=0)]
        attributions2show_per_method_cat = torch.cat(attributions2show_per_method, dim=0)
        if show_only_attr:
            if images.size(1)==3 and attributions2show_per_method_cat.size(1)==1:
                if show_color:
                    attributions2show += [attributions2show_per_method_cat.repeat_interleave(repeats=3, dim=1).abs()*torch.cat(images2show, dim=0)]
                else:
                    attributions2show += [attributions2show_per_method_cat.repeat_interleave(repeats=3, dim=1)]
            else:
                if show_color and xai_method not in 'dxai':
                    attributions2show += [attributions2show_per_method_cat.abs()*torch.cat(images2show, dim=0)]
                else:
                    attributions2show += [attributions2show_per_method_cat]
            im_name = classifier_type+'_attributions' if not show_color else classifier_type+'_attributions_color'
        else:
            attributions2show += [make_anomaly_heatmap(attributions2show_per_method_cat,
                                                    torch.cat(images2show, dim=0))]
            im_name = classifier_type+'_heatmaps'
        if one_heatmap_image:
            attributions2show += [attributions2show_per_method_cat.repeat_interleave(repeats=3, dim=1)]
        # attributions2show += [attributions2show_per_method_cat.repeat_interleave(repeats=3, dim=1)]
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
        col = 3 if one_heatmap_image else (len(images2show) * batch_size)
       
        save_image(torch.cat(attributions2show, dim=0), col,
                   out_folder + os.sep + str(args.resume_iter)+'_'+str((labels2xai[0] == labels[0]).item()) + '_' + im_name + '_' + str(index+1) + '.png')
        images2show = []
        attributions2show_per_method = []
        attributions2show = [] if one_heatmap_image else attributions2show
    return images2show, attributions2show_per_method, attributions2show


def calc_attributions_accuracy(images, attributions, labels, details_dict, xai_method, nets, resnet_classifier, args, att_classifier_type, correct_att, total_att):
    
    if attributions.abs().max()<0.1 and xai_method not in 'dxai':
        attributions = attributions / (attributions.abs().max() + 1e-8)
    
    if args.img_channels==3 and attributions.size(1)==1:
        attributions2class = attributions.repeat_interleave(repeats=3, dim=1)
    else:
        attributions2class = attributions
    if xai_method not in 'dxai':
        attributions2class = images*attributions2class.abs()
    
    outputs = nets.discriminator(attributions2class) if 'resnet' not in att_classifier_type else resnet_classifier(attributions2class)
    _, predicted = torch.max(outputs, dim=1)
    correct_att += (predicted == labels).sum().item()
    total_att += attributions.size(0) 
    details_dict[xai_method].att_accuracy = np.round(1e4 * correct_att / total_att) / 1e4 
    return details_dict, correct_att, total_att


def save_attributions_for_fid(images, attributions, args, classifier_type, xai_method, out_folder, iteration, batch_size, labels, classes):
    N = images.size(0)
    folder_name = out_folder + os.sep + str(args.resume_iter)+'_'+classifier_type+'_attributions' + os.sep + xai_method
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    for k in range(N):
        image_name = str(iteration*batch_size + (k+1)) + '_' + classes[labels[k]] +'.png'
        if attributions[k].abs().max()<0.1 and xai_method not in 'dxai':
            attributions[k] = attributions[k] / (attributions[k].abs().max() + 1e-8)
        if xai_method in 'dxai':
            save_image(attributions[k], ncol=1, filename=folder_name+os.sep+image_name)
        else:
            save_image(images[k]*attributions[k].abs(), ncol=1, filename=folder_name+os.sep+image_name)

        
def print_dictionary(details_dict):
    print('#####################')
    for key in details_dict.keys():
        print(key+': ')
        print(details_dict[key].betas)
        print(details_dict[key].values)
    print('#####################')


def boot_method(details_dict, xai_method, global_beta, num_of_classes=2):
    print('check method creation')
    details_dict.update({xai_method: method_values(num_of_classes=num_of_classes)})
    if 0 not in details_dict[xai_method].betas:
        details_dict = update_beta_and_value(details_dict, xai_method, beta=0)
    beta0_accuracy = None
    for key in details_dict.keys():
        if details_dict[key].values[details_dict[key].betas.index(0)] is not None:
            beta0_accuracy = details_dict[key].values[details_dict[key].betas.index(0)]
            details_dict[xai_method].probs_hists[details_dict[xai_method].betas.index(0)] = details_dict[key].probs_hists[details_dict[key].betas.index(0)]
            break
    details_dict[xai_method].values[details_dict[xai_method].betas.index(0)] = beta0_accuracy
    print_dictionary(details_dict)
    return details_dict


def update_beta_and_value(details_dict, key, beta):
    details_dict[key].betas.append(beta)
    details_dict[key].betas.sort()
    beta_index = details_dict[key].betas.index(beta)
    details_dict[key].values.insert(beta_index, None)
    details_dict[key].probs_hists.insert(beta_index, None)
    return details_dict


def compute_AUC(details_dict, out_folder, args, classifier_type, iters_num, global_beta, T=1):
    if global_beta:
        experiment_name = 'global_beta'
    else:
        experiment_name = 'faithfulness'
    filename = out_folder + os.sep + str(args.resume_iter)+'_AUC_'+experiment_name+'_'+classifier_type+'_'+str(iters_num)+'_iters.txt'
    with open(filename, 'w') as f:
        f.write('####### ' + experiment_name + ' AUC until T='+str(T)+' ##########\n')
        auc_dict = {}
        for key in details_dict.keys():
            betas = np.array(details_dict[key].betas)
            values = np.array(details_dict[key].values)
            n = details_dict[key].values[details_dict[key].betas.index(0)]
            values = values[betas <= T]
            betas = betas[betas <= T]
            AUC = np.trapz(values, betas) / (n+1e-8) 
            if global_beta:
               AUC = AUC / (max(betas)+1e-8)  
            auc_dict.update({key: AUC})
        auc_dict = dict(sorted(auc_dict.items(), key=lambda x:x[1]))
        for key in auc_dict.keys():
            space = ' '
            arrow  = '<-'
            f.write(key+': ' + (20-len(key))*space + str(np.round(1e3*auc_dict[key])/1e3) + str(('dxai' in key)*arrow))
            f.write('\n')
        f.write('######################################\n')
        for key in auc_dict.keys():
            f.write(key+': ')
            for zz in zip(details_dict[key].betas, details_dict[key].values):
                f.write(str(zz))
        f.write('######################################\n')
    f.close()


def load_classifier(classifier_type, data_name, args, num_of_classes, device, nets=None):
    if 'discriminator' not in classifier_type:
        classifier = load_pretrained_classifier(classifier_type, data_name, args.img_channels, args.img_size, num_of_classes, args.classifier_weights_path)
        classifier.to(device)
    else:
        classifier = nets.discriminator
    return classifier

'''
def print_entropy_detatils(y_true, y_pred, entropy_list):
    cf_matrix = np.round(1e3*confusion_matrix(y_true, y_pred, normalize='true'))/1e3
    if cf_matrix.shape[0]<=3:
        print('confusion matrix: ')
        print(cf_matrix)
    entropy_list = torch.stack(entropy_list)
    print('entropy mean:     ', entropy_list.mean(0).cpu().numpy())
    print('entropy std:      ', entropy_list.std(0).cpu().numpy())
'''

def slic2tensor(images, n_segments=256):
    N, C, H, W = images.size()
    features = torch.zeros_like(images)
    for ii in range(N):
        nd_image = tensor2ndarray255(images[ii], denormalize=True)
        segments = slic(nd_image.astype('double'), n_segments=n_segments, slic_zero=True)
        tensor_segments = torch.from_numpy(segments).unsqueeze(0).to(images.device)
        if ii > 0:
            tensor_segments += features[ii-1].max() + 1
        if C == 3:
            tensor_segments = tensor_segments.repeat_interleave(3, 0)
        features[ii] = tensor_segments
    return features.long()
    

def get_last_resume_iter(path, return_all_iters=False):
    max_resume_iter = 0
    iters_list = []
    for file in os.listdir(path):
        iter = int(file[0:6])
        iters_list.append(iter)
        if 'ema' in file and iter >= max_resume_iter:
            max_resume_iter = iter
    iters_list.sort()
    if return_all_iters:
        return max_resume_iter, iters_list
    else:
        return max_resume_iter
