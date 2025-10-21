import torch
from torch import nn


class AdaLN(nn.Module):
    def __init__(
        self, 
        dim, 
    ):
        super().__init__()
        self.ada_map = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(dim, dim * 2)
        )
        self.ln = nn.LayerNorm(
            dim, 
            elementwise_affine=False
        )

    def forward(self, x, y):
        gamma, mu = self.ada_map(y).chunk(2, dim=-1)
        return (1 + gamma) * self.ln(x) + mu


def initialize_adaln_weights(network, std=None):
    if std is not None:
        print("[INFO] Using normal initialization for AdaLN with std={}".format(std))
    for m in network.modules():
        if isinstance(m, nn.Linear):
            if std is not None:
                nn.init.normal_(m.weight, std=std)
            else:
                # print("[INFO] Using Xavier initialization for AdaLN.")
                gain = nn.init.calculate_gain(nonlinearity='relu')
                nn.init.xavier_normal_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)
