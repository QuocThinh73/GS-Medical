import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecularMLP(nn.Module):
    def __init__(self, spec_dim=32, hidden_dim=64, use_time=True):
        super().__init__()
        in_dim = spec_dim + 3  # embedding + view_dir
        if use_time:
            in_dim += 1

        self.use_time = use_time
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, spec_embed, view_dir, time_scalar=None):
        """
        spec_embed: [N, spec_dim]
        view_dir:   [N, 3]
        time_scalar:[N, 1] or None
        """
        x = [spec_embed, view_dir]
        if self.use_time:
            x.append(time_scalar)
        x = torch.cat(x, dim=-1)

        rgb_res = self.net(x)
        return rgb_res