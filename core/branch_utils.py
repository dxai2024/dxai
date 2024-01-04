import torch
import torch.nn as nn


class BranchedNetwork(nn.Module):
    def __init__(self, sub_networks, num_branches, img_channels=3):
        super().__init__()
        self.sub_networks = sub_networks
        self.split = Split(num_split=num_branches)
        self.aggregate = Aggregate(img_channels=img_channels)

    def forward(self, x, s_columnstack, out_features=False):
        Phi, Res = self.split(x)
        almost_Psi, features = self.sub_networks(Phi, s_columnstack, out_features)
        
        masks = 0*almost_Psi.clone().detach()
        
        x_fake, Psi = self.aggregate(almost_Psi)
        if len(features) > 0:
            return x_fake, Psi, features, Res, masks
        else:
            return x_fake, Psi, Phi, Res, masks


class Split(nn.Module):
    def __init__(self, num_split=9):
        super().__init__()
        self.num_split = num_split

    def forward(self, x):
        Phi = x.repeat([1, self.num_split, 1, 1])
        Res = torch.zeros_like(x)
        return Phi, Res


class Aggregate(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.img_channels = img_channels

    def forward(self, Psis):
        y = sum_groups(Psis, phi_depth=self.img_channels)
        return y, Psis


def sum_groups(Phis, phi_depth):
    """
    phi_depth is the number of channels
    """
    num_psi = round(Phis.shape[1] / phi_depth)
    Phis_reshaped = split_groups_along_new_dimension(Phis, num_psi)  # [B, C, H, W, psi_num]
    return torch.sum(Phis_reshaped, dim=-1)


def split_groups_along_new_dimension(Phis, num_phi):
    if num_phi>1: 
        Phis_tuple = torch.chunk(Phis[:, :, :, :, None], num_phi, dim=1)  # chunk along channel dimension
        Phis_cat = torch.cat(Phis_tuple, dim=len(Phis.shape))  # [B, C, H, W, psi_num] concatenate psi along new dimension
    else:
        Phis_cat = Phis.unsqueeze(-1)
    return Phis_cat
