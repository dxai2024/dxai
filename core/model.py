"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
from munch import Munch
import numpy as np
from core.branch_utils import *
import torchvision
import math
import torch.nn.functional as F


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out,
                 normalize=False, downsample=False, group_num=1, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.group_num = group_num
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1, groups=self.group_num)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1, groups=self.group_num)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, groups=self.group_num, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        if s is not None:
            h = self.fc(s)
            h = h.view(h.size(0), h.size(1), 1, 1)
            gamma, beta = torch.chunk(h, chunks=2, dim=1)
            return (1 + gamma) * self.norm(x) + beta
        else:
            return self.norm(x)
            

class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 upsample=False, group_num=1, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.group_num = group_num
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1, groups=self.group_num)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1, groups=self.group_num)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, groups=self.group_num, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


def cat_layers_per_branch(x, y, num_branches):
    Nx, B_Cx ,Hx, Wx = x.size()
    Ny, B_Cy, Hy, Wy = y.size()

    assert Hx == Hy and Wx == Wy

    Cx = int(B_Cx / num_branches)
    Cy = int(B_Cy / num_branches)

    x = x.permute((0, 2, 3, 1)).reshape(Nx, Hx, Wx, num_branches, Cx)
    y = y.permute((0, 2, 3, 1)).reshape(Nx, Hx, Wx, num_branches, Cy)
    z = torch.cat((x, y), dim=4)
    z = z.reshape(Nx, Hx, Wx, num_branches * (Cx + Cy)).permute(0, 3, 1, 2)
    return z


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2, num_style_vecs_per_class=1,
                 deterministic_stylevec=False):
        super().__init__()
        self.deterministic_stylevec = deterministic_stylevec
        self.latent_dim = latent_dim
        linear_dim = 512
        if self.deterministic_stylevec:
            self.unshared = nn.ModuleList()
            for _ in range(num_domains):
                self.unshared += [nn.Linear(latent_dim, style_dim * num_style_vecs_per_class)]
        else:
            layers = []
            layers += [nn.Linear(latent_dim, linear_dim)]
            layers += [nn.ReLU()]
            for _ in range(3):
                layers += [nn.Linear(linear_dim, linear_dim)]
                layers += [nn.ReLU()]
            self.shared = nn.Sequential(*layers)

            self.unshared = nn.ModuleList()
            for _ in range(num_domains):
                self.unshared += [nn.Sequential(nn.Linear(linear_dim, linear_dim),
                                                nn.ReLU(),
                                                nn.Linear(linear_dim, linear_dim),
                                                nn.ReLU(),
                                                nn.Linear(linear_dim, linear_dim),
                                                nn.ReLU(),
                                                nn.Linear(linear_dim, style_dim * num_style_vecs_per_class))]

    def forward(self, z, y):
        if self.deterministic_stylevec:
            h = torch.ones(self.latent_dim).to(z.device)
        else:
            h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(z.device)
        # print('y')
        # print(y.size())
        s_columnstack = out[
            idx, y]  # (batch, style_dim), because y selects the (single) domain for each sample in batch domain
        return s_columnstack


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, img_channels=3, max_conv_dim=512, mean_invariance=False):
        super().__init__()
        self.mean_invariance = mean_invariance
        self.num_domains = num_domains

        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [nn.Conv2d(img_channels, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        self.curr_img_size = img_size
        for ii in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
            self.curr_img_size = self.curr_img_size // 2

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        self.curr_img_size = self.curr_img_size - 4 + 1
        print(self.curr_img_size)
        if self.curr_img_size > 1:
            blocks += [nn.LeakyReLU(0.2)]
            blocks += [nn.Conv2d(dim_out, dim_out, self.curr_img_size, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, y=None):
        if self.mean_invariance:
            x_mean = torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True)
            # print(x_mean)
            x = x - x_mean
        out = self.main(x)
        # if self.curr_img_size>1:
        #    out = nn.functional.avg_pool2d(out,  self.curr_img_size)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        assert out.size(1) == self.num_domains
        # out = self.softmax(out)
        idx = torch.LongTensor(range(x.size(0))).to(x.device)
        out = out[idx, :]
        if y is None:
            return out  # self.softmax(out)
        else:
            out_real_fake = out[idx, y]  # (batch),because y selects the (single) domain for each sample in batch domain
            return out_real_fake, out  # self.softmax(out)


class GroupedGenerator(nn.Module):
    def __init__(self, img_size, num_branches, branch_dimin, style_dim, w_hpf, img_channels=3, style_per_block=True,
                 max_conv_dim=504):

        super().__init__()

        dim_in = num_branches * branch_dimin  # so that it is divisible by 3 AND by (-1+step_num). originally 2**14 // img_size, which is  64
        self.style_dim = style_dim
        self.img_size = img_size
        self.num_branches = num_branches

        group_num = num_branches
        phi_depth = img_channels * num_branches  # img_channels=3 for RGB
        self.from_phis = nn.Sequential(
            nn.InstanceNorm2d(phi_depth, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(phi_depth, dim_in, 3, 1, 1, groups=group_num))

        self.to_3channels = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.Tanh(),
            nn.Conv2d(dim_in, phi_depth, 1, 1, 0, groups=group_num))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - int(np.log2(16))
        self.repeat_num = repeat_num
        self.skip_degree = 0
        self.style_per_block = style_per_block
        self.num_AdaInResBlks = repeat_num + 2  # +2 because of the additional 2 "bottleneck blocks" created below
        if self.style_per_block:
            self.num_style_vecs_per_class = self.num_AdaInResBlks
        else:
            self.num_style_vecs_per_class = 1

        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()

        if w_hpf > 0:
            repeat_num += 1
        for ii in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            if ii < 3:
                self.encode.append(nn.Sequential(
                    ResBlk(dim_in, dim_in, normalize=True, downsample=False, group_num=group_num),
                    ResBlk(dim_in, dim_in, normalize=True, downsample=False, group_num=group_num),
                    ResBlk(dim_in, dim_in, normalize=True, downsample=False, group_num=group_num),
                    ResBlk(dim_in, dim_out, normalize=True, downsample=True, group_num=group_num)))
                #
            else:
                self.encode.append(nn.Sequential(
                    ResBlk(dim_in, dim_in, normalize=True, downsample=False, group_num=group_num),
                    ResBlk(dim_in, dim_out, normalize=True, downsample=True, group_num=group_num)))

            if ii >= self.skip_degree:
                self.decode.insert(0, AdainResBlk(2 * dim_out, dim_in, style_dim,
                                                  w_hpf=w_hpf, upsample=True, group_num=group_num)
                                   )  # stack-like
            else:
                self.decode.insert(0, AdainResBlk(dim_out, dim_in, style_dim,
                                                  w_hpf=w_hpf, upsample=True, group_num=group_num)
                                   )  # stack-like
            # self.decode.insert(0 , my_squential(dim_in, dim_out, style_dim, w_hpf, group_num))
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True, group_num=group_num))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf, group_num=group_num))

    def forward(self, x, s_columnstack, out_features=False):
        z_list = []
        features = []
        x = self.from_phis(x)

        s_tuple = torch.chunk(s_columnstack, chunks=self.num_style_vecs_per_class, dim=1)

        for block_idx, block in enumerate(self.encode):
            x = block(x)
            if block_idx < self.repeat_num and block_idx >= self.skip_degree:
                z_list.insert(0, x)

        for block_idx, block in enumerate(self.decode):

            if self.style_per_block:
                s = s_tuple[block_idx]
            else:
                s = s_tuple[0]

            if block_idx >= 2 and block_idx < len(self.decode) - self.skip_degree:
                x = cat_layers_per_branch(z_list[block_idx - 2], x, self.num_branches)
            x = block(x, s)
            if out_features:
                features.append(x)
        if out_features:
            features = z_list + features
        if out_features:
            return self.to_3channels(x), features
        else:
            return self.to_3channels(x), []  # This is psi!


class Classifier(nn.Module):
    def __init__(self, img_size=256, num_domains=2, img_channels=3, max_conv_dim=512, mean_invariance=False):
        super().__init__()
        self.mean_invariance = mean_invariance
        dim_in = 2 ** 14 // img_size
        blocks = []
        blocks += [nn.Conv2d(img_channels, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0,)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        if self.mean_invariance:
            x_mean = torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True)
            x = x - x_mean
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        out = self.softmax(out)
        return out


class Classifier2(nn.Module):
    def __init__(self, img_size=256, num_branches=2, img_channels=3, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = num_branches * 2**14 // img_size
        max_conv_dim = num_branches * max_conv_dim
        self.pool = [nn.MaxPool2d(kernel_size=2) for i in range(6)]  # nn.AvgPool2d(kernel_size=2)
        self.pool0 = self.pool[0]
        self.pool1 = self.pool[1]
        self.pool2 = self.pool[2]
        self.pool3 = self.pool[3]
        self.pool4 = self.pool[4]
        self.pool5 = self.pool[5]
        self.actv = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=dim_in, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=dim_in, out_channels=2*dim_in, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=2*dim_in, out_channels=4*dim_in, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=4*dim_in, out_channels=4*dim_in, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=4*dim_in, out_channels=8*dim_in, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=8*dim_in, out_channels=8*dim_in, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=8*dim_in, out_channels=8*dim_in, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=8*dim_in, out_channels=8*dim_in, kernel_size=3, padding=1)
        self.conv4x4 = nn.Conv2d(in_channels=8*dim_in, out_channels=num_domains, kernel_size=int(img_size/2**6), padding=0)
        #self.FC = nn.Linear(in_features=16*64*64, out_features=16*64*64)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)  # (256,256)
        x = self.actv(x)
        x = self.conv2(x)
        x = self.pool0(x)
        x = self.actv(x)
        x = self.conv3(x)  # (128,128)
        x = self.pool1(x)
        x = self.actv(x)
        x = self.conv4(x)  # (64,64)
        x = self.pool2(x)
        x = self.actv(x)
        x = self.conv5(x)  # (32,32)
        x = self.pool3(x)
        x = self.actv(x)
        x = self.conv6(x)  # (16,16)
        x = self.pool4(x)
        x = self.actv(x)
        x = self.conv7(x)  # (8,8)
        x = self.pool5(x)
        x = self.actv(x)
        x = self.conv8(x)  # (4,4)
        x = self.conv4x4(x)  #(1,1)
        x = x.squeeze(-1).squeeze(-1)
        # x = self.softmax(x)
        return x


class Classifier3(nn.Module):
    def __init__(self, img_size=256, num_branches=2, img_channels=3, num_domains=2, max_conv_dim=512):
        super().__init__()
        self.num_domains = num_domains
        dim_in = num_branches * 2**14 // img_size
        max_conv_dim = num_branches * max_conv_dim
        blocks = []
        blocks += [nn.Conv2d(img_channels, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        curr_img_size = img_size
        for ii in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [nn.ReLU()]
            blocks += [nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=3, padding=1)]
            blocks += [nn.MaxPool2d(kernel_size=2)]
            dim_in = dim_out
            curr_img_size = curr_img_size // 2

        blocks += [nn.ReLU()]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        curr_img_size = curr_img_size - 4 + 1
        print(curr_img_size)
        if curr_img_size > 1:
            blocks += [nn.ReLU()]
            blocks += [nn.Conv2d(dim_out, dim_out, curr_img_size, 1, 0)]
        blocks += [nn.ReLU()]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        assert out.size(1) == self.num_domains
        return out


class simple_classifier(nn.Module):
    def __init__(self, img_size=256, img_channels=3, num_domains=2, dim_in=16, max_conv_dim=64):
        super().__init__()
        self.num_domains = num_domains
        kernel_size = int(max(2, np.round(np.sqrt(img_size/32))))
        print('kernel_size: ', kernel_size)
        blocks = []
        blocks += [nn.Conv2d(img_channels, dim_in, 3, 1, 1)]
        blocks += [nn.MaxPool2d(kernel_size=kernel_size)]
        curr_img_size = img_size // kernel_size
        blocks += [nn.ReLU()]
        dim_out = min(dim_in * 2, max_conv_dim)
        blocks += [nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=3, padding=1)]
        blocks += [nn.MaxPool2d(kernel_size=kernel_size)]
        curr_img_size = curr_img_size // kernel_size
        self.main = nn.Sequential(*blocks)
        print('curr_img_size: ', curr_img_size)
        print('dim_out: ', dim_out)
        self.fc1 = nn.Linear(in_features=curr_img_size*curr_img_size*dim_out, out_features=128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=num_domains)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)  # (batch, 128)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(x.size(0), -1)  # (batch, num_domains)
        assert x.size(1) == self.num_domains
        return x


def load_pretrained_classifier(classifier_type, data_name, img_channels, img_size, num_of_classes):
    if 'classifier2' in classifier_type:
        classifier_path = './'+data_name+'_classifier2'+ '_ch_' + str(img_channels) + '_weights.ckpt'
        classifier = Classifier2(img_size=img_size, num_branches=1, img_channels=img_channels,
                                    num_domains=num_of_classes)
    elif 'classifier3' in classifier_type:
        classifier_path = './'+data_name+'_classifier3'+ '_ch_' + str(img_channels) + '_weights.ckpt'
        classifier = Classifier3(img_size=img_size, num_branches=1, img_channels=img_channels,
                                    num_domains=num_of_classes)
    elif 'simple_classifier' in classifier_type:
        classifier_path = './' + data_name + '_simple_classifier' + '_ch_' + str(img_channels) + '_weights.ckpt'
        classifier = simple_classifier(img_size=img_size, img_channels=img_channels, num_domains=num_of_classes)
    elif 'resnet18' in classifier_type:
        classifier_path = './'+data_name+'_resnet18' + '_ch_' + str(img_channels) + '_weights.ckpt'
        classifier = torchvision.models.resnet18(pretrained=False)
        nr_filters = classifier.fc.in_features  # number of input features of last layer
        classifier.fc = nn.Linear(nr_filters, num_of_classes)
        classifier.conv1 = nn.Conv2d(img_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif classifier_type in 'vgg11':
        classifier_path = './'+data_name+'_vgg11' + '_ch_' + str(img_channels) + '_weights.ckpt'
        classifier = torchvision.models.vgg11(pretrained=False)
        classifier.classifier[6] = nn.Linear(in_features=4096, out_features=num_of_classes)
        classifier.features[0] = nn.Conv2d(img_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    classifier.load_state_dict(torch.load(classifier_path))

    return classifier.eval()


def build_model(args):

    branched_net = GroupedGenerator(img_size=args.img_size, num_branches=args.num_branches,
                                    branch_dimin=args.branch_dimin, style_dim=args.style_dim,
                                    w_hpf=args.w_hpf, img_channels=args.img_channels,
                                    style_per_block=args.style_per_block, max_conv_dim=args.num_branches*64)

    generator = BranchedNetwork(branched_net, args.num_branches, img_channels=args.img_channels)

    generator = nn.DataParallel(generator)
    num_style_vecs_per_class = generator.module.sub_networks.num_style_vecs_per_class

    discriminator = Discriminator(args.img_size, args.num_domains, args.img_channels, max_conv_dim=512,
                                      mean_invariance=False)

    mapping_network = MappingNetwork(args.latent_dim, style_dim=args.style_dim, num_domains=args.num_domains,
                                     num_style_vecs_per_class=num_style_vecs_per_class, deterministic_stylevec=False)

    mapping_network = nn.DataParallel(mapping_network)
    discriminator = nn.DataParallel(discriminator)

    discriminator_ema = copy.deepcopy(discriminator)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)

    if args.use_pretrained_classifier:
        classifier = load_pretrained_classifier(classifier_type=args.classifier_type, data_name=args.data_name,
                                                img_channels=args.img_channels, img_size=args.img_size, num_of_classes=args.num_domains)
        classifier = nn.DataParallel(classifier)
        classifier_ema = copy.deepcopy(classifier)
        nets = Munch(generator=generator,
                     mapping_network=mapping_network,
                     discriminator=discriminator,
                     classifier=classifier)
        nets_ema = Munch(generator=generator_ema,
                         mapping_network=mapping_network_ema,
                         discriminator=discriminator_ema,
                         classifier=classifier_ema)
    else:
        nets = Munch(generator=generator,
                     mapping_network=mapping_network,
                     discriminator=discriminator)
        nets_ema = Munch(generator=generator_ema,
                         mapping_network=mapping_network_ema,
                         discriminator=discriminator_ema)

    return nets, nets_ema

