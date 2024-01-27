"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from types import SimpleNamespace
import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from core.utils import my_gradient
from core.branch_utils import sum_groups


class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Check and update the train_img_dir in args
        if args.train_img_dir in '../Data/train':
            args.train_img_dir = '../Data/' + args.data_name + '/train'
        args.num_domains = len(next(os.walk(args.train_img_dir))[1])
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build the model using the provided arguments
        self.nets, self.nets_ema = build_model(args)

        # Set networks as attributes of Solver
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            # Initialize optimizers for different networks
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan' or net == 'classifier':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if (
                            net == 'mapping_network' or net == 'filter_network' or net == 'rec_filter_network') else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            # Initialize checkpoint IO for saving and loading models
            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), **self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]
        else:
            # Initialize checkpoint IO for loading ema models during testing
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema)]

        # Move the entire Solver to the specified device
        self.to(self.device)
        # Initialize network parameters using he_init
        for name, network in self.named_children():
            if ('ema' not in name) and ('fan' not in name) and ('classifier' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        # Save the current state of networks and optimizers
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        # Load the state of networks and optimizers from a saved checkpoint
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        # Reset gradients of all optimizers
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders, test_loaders=None):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # Fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, None, args.latent_dim, 'train')

        # Resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # Fetch images and labels for training
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src
            y_trg = torch.randint(low=0, high=args.num_domains, size=(x_real.shape[0],)).to(y_org.device)

            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2

            if args.alpha_blend:
                # Alpha blending for domain mixing
                y_trg = y_org.clone().long()
                x_real = x_real.repeat(2, 1, 1, 1)
                y_org = y_org.repeat(2).long()
                if args.num_domains == 2:
                    y_trg = torch.cat((y_trg, 1 - y_trg)).long().to(x_real.device)
                else:
                    y_other = args.num_domains * torch.ones_like(y_trg)
                    for jj in range(y_trg.size(0)):
                        class_list = list(range(0, args.num_domains))
                        class_list.remove(int(y_trg[jj].detach().cpu().numpy()))
                        y_other[jj] = random.choice(class_list)
                    y_trg = torch.cat((y_trg, y_other)).long().to(x_real.device)
                z_trg = torch.randn(x_real.size(0), args.latent_dim).to(x_real.device)
                alpha_vec = torch.rand((x_real.size(0) // 2, args.num_branches - 1)).to(x_real.device)
                alpha_vec = alpha_vec.repeat_interleave(repeats=args.img_channels, dim=1).unsqueeze(-1).unsqueeze(-1)
            else:
                alpha_vec = None

            # Train the discriminator
            d_loss, d_losses_latent1 = compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=z_trg,
                                                      alpha_vec=alpha_vec)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # Train the generator
            g_loss, g_losses_latent1 = compute_g_loss(nets, args, x_real, y_trg, z_trgs=[z_trg, z_trg2],
                                                     y_org=y_org, alpha_vec=alpha_vec, iteration=i)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()

            # Compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.discriminator, nets_ema.discriminator, beta=0.999)


            # print out log info
            if i % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.total_iters)
                all_losses = dict()
                latent_list = [d_losses_latent1, g_losses_latent1]
                G_D_names = ['Dz/', 'Gz/']
                for loss, prefix in zip(latent_list,
                                        G_D_names):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # save model checkpoints
            if i % args.save_every == 0:
                print('saving checkpoint..')
                self._save_checkpoint(step=i + 1)

            sample_every = args.sample_every
            # generate images for debugging
            if i % sample_every == 0:
                step = i + 1
                src = next(InputFetcher(test_loaders.src, None, args.latent_dim, 'test'))
                # ref = next(InputFetcher(test_loaders.ref, None, args.latent_dim, 'test'))
                os.makedirs(args.sample_dir, exist_ok=True)
                inputs = {'x_src': src.x, 'y_src': src.y}
                inputs = SimpleNamespace(**inputs)
                utils.debug_image(nets_ema, args, inputs=inputs, step=step)


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None, alpha_vec=[]):
    # Compute discriminator loss
    x_real.requires_grad_()
    
    # Discriminator forward pass with real input
    out_real_fake, out = nets.discriminator(x_real, y_org)
    
    # Adversarial loss for real input
    loss_real = adv_loss(out_real_fake, 1)
    
    # Classification loss for real input
    if args.use_pretrained_classifier:
        with torch.no_grad():
            classifier_out = nets.classifier(x_real)
            classifier_out = F.softmax(classifier_out, dim=1)
            _, predicted = torch.max(classifier_out, dim=1)
        _, out = nets.discriminator(x_real, predicted)
        out = F.log_softmax(out, dim=1)
        loss_class_real = F.kl_div(out, classifier_out, reduction='batchmean')
    else:
        loss_class_real = F.cross_entropy(out / args.softmax_temp, y_org)
    
    # R1 regularization term
    loss_reg = r1_reg(out_real_fake, x_real)

    # Generate fake output
    with torch.no_grad():
        s_trg = nets.mapping_network(z_trg, y_trg)
        x_fake, Psi, Phi, Res, Masks = nets.generator(x_real, s_trg)
        if args.alpha_blend:
            if args.zero_st:
                Psi[y_trg != y_org, 0:args.img_channels, :, :] = 0 * Psi[y_trg != y_org, 0:args.img_channels, :, :]
            x_fake = alpha_superposition(Psi, Res, alpha_vec, args)

    # Discriminator forward pass with fake input
    out_real_fake, out = nets.discriminator(x_fake, y_trg)

    # Adversarial loss for fake input
    loss_fake = adv_loss(out_real_fake, 0)

    # Classification loss for fake input
    loss_class_fake = F.cross_entropy(out / args.softmax_temp, y_trg)

    # Overall loss
    loss = loss_real + loss_fake + 1 * loss_class_real + 0 * loss_class_fake + args.lambda_reg * loss_reg

    # Return the computed losses
    return loss, Munch(real=loss_real.item(), fake=loss_fake.item(),
                       reg=loss_reg.item(), class_real=loss_class_real.item(), class_fake=loss_class_fake.item())


def compute_g_loss(nets, args, x_real, y_trg, z_trgs=None, y_org=None, alpha_vec=[], iteration=0):
    # Compute generator loss
    if z_trgs is None:
        s_trg = nets.style_encoder(x_real, y_trg)
    else:
        z_trg, z_trg2 = z_trgs
        s_trg = nets.mapping_network(z_trg, y_trg)

    # Generate fake output
    x_fake, Psi, Phi, Res, Masks = nets.generator(x_real, s_trg, out_features=args.out_features)
    if args.alpha_blend:
        if args.zero_st:
            Psi[y_trg != y_org, 0:args.img_channels, :, :] = 0 * Psi[y_trg != y_org, 0:args.img_channels, :, :]
        half_B = int(x_real.shape[0] / 2)
        x_fake = alpha_superposition(Psi, Res, alpha_vec, args)

    # Discriminator forward pass with fake input
    if args.use_pretrained_classifier:
        out_real_fake, _ = nets.discriminator(x_fake, y_trg)
        out = nets.classifier(x_fake)
    else:
        out_real_fake, out = nets.discriminator(x_fake, y_trg)

    # Adversarial loss
    loss_adv = adv_loss(out_real_fake, 1)

    # Classification loss for fake input
    loss_class_fake = F.cross_entropy(out / args.softmax_temp, y_trg)

    # Additional losses if alpha_blend is enabled
    if args.alpha_blend:
        if z_trgs is not None:
            # Similarity losses
            l_sim = sim_loss(x_fake, x_real, y_trg, y_org, args.img_channels)
            l_grad_sim = grad_loss(x_fake[y_trg == y_org], x_real[y_trg == y_org], order=1)
            l_ano_sim = anomaly_sim_loss(x_fake[y_trg == y_org], x_real[y_trg == y_org], Psi[y_trg == y_org], args)

            # Orthogonal loss for network parameters if out_features is enabled
            if args.out_features:
                l_params_corr = features_orthogonal_loss(Phi, Psi.device, args.num_branches)
            else:
                l_params_corr = torch.tensor(0).to(Psi.device).float()

            # Overall loss with additional losses
            loss = args.lambda_adv * loss_adv + args.lambda_class_fake * loss_class_fake + \
                   args.lambda_rec * l_sim + args.lambda_grad_rec * l_grad_sim + args.lambda_dis_rec * l_ano_sim + \
                   args.lambda_orth * l_params_corr

            # Return the computed losses and additional information
            loss_latent = Munch(adv=loss_adv.item(), class_fake=loss_class_fake.item(), sim=l_sim.item(),
                                grad=l_grad_sim.item(), ano_sim=l_ano_sim.item(), l_params_corr=l_params_corr.item())
        else:
            loss = args.lambda_adv * loss_adv + args.lambda_class_fake * loss_class_fake

            # Return the computed losses
            loss_latent = Munch(adv=loss_adv.item(), class_fake=loss_class_fake.item())
    else:
        # Loss without additional terms for alpha_blend
        sim_diff = x_fake - x_real
        l_sim = (sim_diff ** 2).mean() + sim_diff.abs().mean()
        loss = args.lambda_adv * loss_adv + args.lambda_rec
        loss_latent = Munch(adv=loss_adv.item(), sim=l_sim.item())

    # Return the computed losses
    return loss, loss_latent



def alpha_superposition(Psi, Res, alpha_vec, args):
    """
    Perform alpha superposition for blending branches in the generator.

    Args:
        Psi : Output from the generator's Psi branch.
        Res : Residual feature maps.
        alpha_vec : Alpha vector for blending.
        args: Additional configuration arguments.

    Returns:
        torch.Tensor: Alpha superposed fake image.

    """
    # Separate the first branch and other branches
    first_branch = Psi[:, 0:args.img_channels, :, :]
    other_branches = Psi[:, args.img_channels::, :, :]
    x_fake = torch.zeros(Psi.size(0), args.img_channels, Psi.size(2), Psi.size(3)).to(Psi.device)
    half_B = int(Psi.shape[0] / 2)

    # Alpha superposition for the first half of the batch
    x_fake[0:half_B] = Res[0:half_B] + (first_branch[0:half_B] +
                        sum_groups(alpha_vec * other_branches[0:half_B], phi_depth=args.img_channels) +
                        sum_groups((1 - alpha_vec) * other_branches[half_B::], phi_depth=args.img_channels))

    # Alpha superposition for the second half of the batch
    x_fake[half_B::] = Res[half_B::] + (first_branch[half_B::] +
                        sum_groups((1 - alpha_vec) * other_branches[0:half_B], phi_depth=args.img_channels) +
                        sum_groups(alpha_vec * other_branches[half_B::], phi_depth=args.img_channels))
    return x_fake


def sim_loss(x_fake, x_real, y_trg, y_org, img_channels):
    """
    Compute similarity loss between fake and real images.

    Args:
        x_fake : Fake images.
        x_real : Real images.
        y_trg : Target domain labels for fake images.
        y_org : Original domain labels for real images.
        img_channels (int): Number of image channels.

    Returns:
        torch.Tensor: Similarity loss.

    """
    sim_diff = x_fake - x_real
    l_sim = (sim_diff ** 2).mean() + sim_diff.abs().mean()
    return l_sim


def anomaly_sim_loss(x_fake, x_real, Psi, args):
    """
    Compute anomaly similarity loss between fake and real images based on Psi values.

    Args:
        x_fake : Fake images.
        x_real : Real images.
        Psi : Psi values from the generator.
        args: Additional configuration arguments.

    Returns:
        torch.Tensor: Anomaly similarity loss.

    """
    with torch.no_grad():
        abs_Psi = Psi[:, 0:args.img_channels].abs()
        abs_Psi = abs_Psi - torch.mean(abs_Psi, dim=(2, 3), keepdim=True)

    sim_diff = x_fake[abs_Psi > 0] - x_real[abs_Psi > 0]
    l_sim = (sim_diff ** 2).mean() + sim_diff.abs().mean()
    return l_sim


def features_orthogonal_loss(features_list, device, num_branches):
    """
    Compute features orthogonal loss for network features.

    Args:
        features_list (list): List of features from different branches.
        device: Device where the computation takes place.
        num_branches (int): Number of branches in the network.

    Returns:
        torch.Tensor: Features orthogonal loss.

    """
    loss = torch.tensor(0).to(device).float()
    counter = 0

    # Iterate through features from different branches
    for pp, features in enumerate(features_list):
        C = int(features.size(1) / num_branches)
        for ii in range(num_branches):
            feature1 = features[:, ii * C:(ii + 1) * C]
            feature1 = feature1.reshape((features.size(0), C, -1))
            feature1 = feature1 - feature1.mean(2, keepdim=True)
            feature1 = feature1 / (feature1.std(2, keepdim=True) + 1e-6)

            for jj in range(ii + 1):
                if ii != jj:
                    feature2 = features[:, jj * C:(jj + 1) * C]
                    feature2 = feature2.reshape((features.size(0), C, -1))
                    feature2 = feature2 - feature2.mean(2, keepdim=True)
                    feature2 = feature2 / (feature2.std(2, keepdim=True) + 1e-6)
                    cov = torch.matmul(feature1, feature2.transpose(1, 2)) / feature1.size(2)
                    loss += (cov**2).mean()

        counter += 1

    return loss / counter



def my_laplacian(v):
    """
    Compute the Laplacian of a given tensor.

    Args:
        v : Input tensor.

    Returns:
        torch.Tensor: Laplacian of the input tensor.

    """
    v_x, v_y = my_gradient(v)
    v_xx, _ = my_gradient(v_x)
    _, v_yy = my_gradient(v_y)
    return v_xx + v_yy


def my_gradient_sum(v):
    """
    Compute the sum of the gradients along x and y directions of a given tensor.

    Args:
        v : Input tensor.

    Returns:
        torch.Tensor: Sum of gradients along x and y directions.

    """
    v_x, v_y = my_gradient(v)
    return v_x + v_y


def grad_loss(x1, x2, order=1):
    """
    Compute gradient loss between two tensors.

    Args:
        x1 : First tensor.
        x2 : Second tensor.
        order (int): Order of the gradient loss (1 or 2).

    Returns:
        torch.Tensor: Computed gradient loss.

    """
    gx1, gy1 = my_gradient(x1)
    gx2, gy2 = my_gradient(x2)
    diff_x = gx1 - gx2
    diff_y = gy1 - gy2

    if order == 2:
        gxx1, _ = my_gradient(gx1)
        _, gyy1 = my_gradient(gy1)
        gxx2, _ = my_gradient(gx2)
        _, gyy2 = my_gradient(gy2)
        diff_xx = gxx1 - gxx2
        diff_yy = gyy1 - gyy2
        l_grad2 = (diff_xx.abs() + diff_yy.abs()).mean() + (diff_xx ** 2 + diff_yy ** 2).mean()
    else:
        l_grad2 = 0

    l_grad = l_grad2 + (diff_x.abs() + diff_y.abs()).mean() + (diff_x ** 2 + diff_y ** 2).mean()
    return l_grad


def moving_average(model, model_test, beta=0.999):
    """
    Update the parameters of a model using a moving average.

    Args:
        model (torch.nn.Module): Original model.
        model_test (torch.nn.Module): Model to be updated.
        beta (float): Exponential decay rate.

    """
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    """
    Compute adversarial loss using binary cross entropy with logits.

    Args:
        logits : Logits from the model.
        target (int): Target label (1 for real, 0 for fake).

    Returns:
        torch.Tensor: Computed adversarial loss.

    """
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    """
    Compute regularization term for zero-centered gradient penalty.

    Args:
        d_out : Output from the discriminator.
        x_in : Input tensor.

    Returns:
        torch.Tensor: Regularization term.

    """
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg
