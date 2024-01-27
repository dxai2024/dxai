import torch
import torch.nn as nn


class BranchedNetwork(nn.Module):
    def __init__(self, sub_networks, num_branches, img_channels=3):
        """
        Branched network module that splits, processes, and aggregates features using sub-networks.

        Args:
            sub_networks (nn.Module): Sub-networks for processing Phi.
            num_branches (int): Number of branches in the network.
            img_channels (int): Number of image channels.

        """
        super().__init__()
        self.sub_networks = sub_networks
        self.split = Split(num_split=num_branches)
        self.aggregate = Aggregate(img_channels=img_channels)

    def forward(self, x, s_columnstack, out_features=False):
        """
        Forward pass through the BranchedNetwork.

        Args:
            x (torch.Tensor): Input tensor.
            s_columnstack (torch.Tensor): Stacked style vectors.
            out_features (bool): Flag indicating whether to output features.

        Returns:
            tuple: Tuple containing generated fake images, Psi values, features, Residuals, and masks.

        """
        Phi, Res = self.split(x)
        almost_Psi, features = self.sub_networks(Phi, s_columnstack, out_features)
        
        masks = 0 * almost_Psi.clone().detach()
        
        x_fake, Psi = self.aggregate(almost_Psi)
        if len(features) > 0:
            return x_fake, Psi, features, Res, masks
        else:
            return x_fake, Psi, Phi, Res, masks


class Split(nn.Module):
    def __init__(self, num_split=9):
        """
        Split module that duplicates the input tensor for each branch.

        Args:
            num_split (int): Number of branches.

        """
        super().__init__()
        self.num_split = num_split

    def forward(self, x):
        """
        Forward pass through the Split module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Tuple containing duplicated Phi and zero tensors (Res).

        """
        Phi = x.repeat([1, self.num_split, 1, 1])
        Res = torch.zeros_like(x)
        return Phi, Res


class Aggregate(nn.Module):
    def __init__(self, img_channels=3):
        """
        Aggregate module that sums the processed branches to create the final output.

        Args:
            img_channels (int): Number of image channels.

        """
        super().__init__()
        self.img_channels = img_channels

    def forward(self, Psis):
        """
        Forward pass through the Aggregate module.

        Args:
            Psis (torch.Tensor): Processed Psi values.

        Returns:
            tuple: Tuple containing aggregated fake images and Psis.

        """
        y = sum_groups(Psis, phi_depth=self.img_channels)
        return y, Psis


def sum_groups(Phis, phi_depth):
    """
    Sum the processed branches along the channel dimension.

    Args:
        Phis (torch.Tensor): Processed Phi values.
        phi_depth (int): Number of channels.

    Returns:
        torch.Tensor: Summed result.

    """
    num_psi = round(Phis.shape[1] / phi_depth)
    Phis_reshaped = split_groups_along_new_dimension(Phis, num_psi)  # [B, C, H, W, psi_num]
    return torch.sum(Phis_reshaped, dim=-1)


def split_groups_along_new_dimension(Phis, num_phi):
    """
    Split the processed branches along a new dimension.

    Args:
        Phis (torch.Tensor): Processed Phi values.
        num_phi (int): Number of branches.

    Returns:
        torch.Tensor: Processed Phis with a new dimension.

    """
    if num_phi > 1:
        Phis_tuple = torch.chunk(Phis[:, :, :, :, None], num_phi, dim=1)  # chunk along channel dimension
        Phis_cat = torch.cat(Phis_tuple, dim=len(Phis.shape))  # [B, C, H, W, psi_num] concatenate psi along new dimension
    else:
        Phis_cat = Phis.unsqueeze(-1)
    return Phis_cat
