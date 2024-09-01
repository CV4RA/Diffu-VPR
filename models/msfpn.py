import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class MSFPN(nn.Module):
    def __init__(self, vlad_clusters=64, dim=128):
        super(MSFPN, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vlad = NetVLAD(num_clusters=vlad_clusters, dim=dim)

    def forward(self, x):
        x = self.vit(x)
        x = self.vlad(x)
        return x

class NetVLAD(nn.Module):
    def __init__(self, num_clusters, dim):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.clusters = nn.Parameter(torch.randn(num_clusters, dim))
        self.linear = nn.Linear(dim, num_clusters)

    def forward(self, x):
        soft_assign = self.linear(x)
        soft_assign = torch.nn.functional.softmax(soft_assign, dim=-1)
        residuals = x.unsqueeze(1) - self.clusters.unsqueeze(0)
        vlad = residuals * soft_assign.unsqueeze(-1)
        vlad = vlad.sum(dim=1)
        vlad = torch.nn.functional.normalize(vlad, dim=-1)
        return vlad
