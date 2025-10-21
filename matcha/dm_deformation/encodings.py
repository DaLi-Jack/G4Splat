import numpy as np
import torch
import torch.nn as nn
from typing import List


class FrequencyPositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_freqs=4):
        super(FrequencyPositionalEncoding, self).__init__()
        self.num_freqs = num_freqs
        self.input_dim = input_dim
        freq_bands = 2. ** torch.linspace(0., num_freqs-1, num_freqs)
        freq_bands = freq_bands * np.pi
        freq_bands = freq_bands.unsqueeze(0).unsqueeze(0)
        self.freq_bands = nn.Parameter(freq_bands, requires_grad=False)
        
    def forward(self, x):
        # x is supposed to be in [0, 1]
        y = x.unsqueeze(-1)
        y = (2 * y - 1) * self.freq_bands  # Moving x to [-freq_bands, freq_bands]
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=-1)
        y = y.view(y.shape[0], -1)
        return y


# TODO
class SphericalHarmonicDirectionalEncoding(nn.Module):
    def __init__(self, input_dim, sh_degree=3):
        super(SphericalHarmonicDirectionalEncoding, self).__init__()
        raise NotImplementedError("This Encoding is not implemented yet.")
        
    def forward(self, x):
        pass
    
    
class LearnableDirectionalEncoding(nn.Module):
    def __init__(self, encoding_dim, num_directions):
        super(LearnableDirectionalEncoding, self).__init__()
        self.num_directions = num_directions
        self.directions = nn.Parameter(
            torch.zeros(num_directions, encoding_dim), 
            requires_grad=True,
        )
        
    def forward(self, idx):
        return self.directions[idx]
    
    
class DepthEncoding(nn.Module):
    def __init__(
        self, 
        num_charts:int,
        num_bins:int,
        encoding_dim:int,
        interpolation_mode:str='bilinear',
        padding_mode:str='border',
        initialization_range=1e-4,
    ):
        super(DepthEncoding, self).__init__()
        self.num_charts = num_charts
        self.num_bins = num_bins
        self.encodings = nn.Parameter(
            initialization_range * (-1. + 2. * torch.rand(num_charts, encoding_dim, num_bins)),  # (n_charts, encoding_dim, num_bins)
            requires_grad=True,
        )
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
    
    def forward(self, depth_coords:torch.Tensor):
        # depth_coords should have either shape (n_charts, n_verts_per_chart) or shape (n_charts, height, width) 
        # and have values in [0, 1].
        assert depth_coords.shape[0] == self.num_charts
        if depth_coords.dim() == 2:
            n_verts_per_chart = depth_coords.shape[1]
        elif depth_coords.dim() == 3:
            n_verts_per_chart = depth_coords.shape[1] * depth_coords.shape[2]
        else:
            raise ValueError("Invalid depth_coords shape.")
        
        n_verts_per_chart = depth_coords.shape[1]
        grid = torch.cat(
            [
                torch.zeros(self.num_charts, n_verts_per_chart, 1, 1, device=depth_coords.device),
                -1 + 2 * depth_coords.view(self.num_charts, n_verts_per_chart, 1, 1),
            ], 
            dim=-1
        )
        return torch.nn.functional.grid_sample(
            input=self.encodings[..., None],  # (n_charts, encoding_dim, num_bins, 1)
            grid=grid,  # (n_charts, n_verts_per_chart, 1, 2)
            mode=self.interpolation_mode,
            padding_mode=self.padding_mode,
        )[..., 0].permute(0, 2, 1).view(self.num_charts, *depth_coords.shape[1:], -1) # (n_charts, n_verts_per_chart, encoding_dim) or (n_charts, height, width, encoding_dim)
    
    
class ChartsEncoding(nn.Module):
    def __init__(
        self, 
        num_charts:int, 
        encoding_h:int, 
        encoding_w:int, 
        encoding_dim:int,
        interpolation_mode:str='bilinear',
        padding_mode:str='border',
        align_corners:bool=False,
        initialization_range=1e-4,
    ):
        super(ChartsEncoding, self).__init__()
        self.num_charts = num_charts
        self.encodings = nn.Parameter(
            initialization_range * (-1. + 2. * torch.rand(num_charts, encoding_dim, encoding_h, encoding_w)), 
            requires_grad=True,
        )
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        
    def forward(self, uv:torch.Tensor):
        # uv should have shape (n_charts, new_h, new_w, 2)
        assert uv.dim() == 4
        assert uv.shape[0] == self.num_charts
        assert uv.shape[-1] == 2
        
        return torch.nn.functional.grid_sample(
            input=self.encodings, 
            grid=uv, 
            mode=self.interpolation_mode, 
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        ).permute(0, 2, 3, 1)  # (n_charts, new_h, new_w, encoding_dim)
        
        
class MultiResChartsEncoding(nn.Module):
    def __init__(
        self,
        num_charts:int, 
        height:int, 
        width:int, 
        resolutions:List[int],
        encoding_dim_per_res:int,
        interpolation_mode:str='bilinear',
        padding_mode:str='border',
        align_corners:bool=False,
        initialization_range=1e-4,
    ):
        super(MultiResChartsEncoding, self).__init__()
        
        self.num_charts = num_charts
        self.height = height
        self.width = width
        self.resolutions = resolutions
        self.encoding_dim = encoding_dim_per_res * len(resolutions)
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.initialization_range = initialization_range
        
        charts_encoding = [
            ChartsEncoding(
                num_charts=num_charts, 
                encoding_h=int(res * height), 
                encoding_w=int(res * width), 
                encoding_dim=encoding_dim_per_res,
                interpolation_mode=interpolation_mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
                initialization_range=initialization_range,
            ) for res in resolutions
        ]
        self.charts_encoding = nn.ModuleList(charts_encoding)
        
    def forward(self, pts_uv):
        return torch.cat(
            [
                chart_encoding(pts_uv) 
                for chart_encoding in self.charts_encoding
            ], 
            dim=-1
        )
