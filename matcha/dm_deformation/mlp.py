import numpy as np
import torch
from torch import nn

from matcha.dm_deformation.encodings import FrequencyPositionalEncoding


def initialize_mlp_weights(network, std=None):
    if std is not None:
        print("[INFO] Using normal initialization for MLP with std={}".format(std))
    for m in network.modules():
        if isinstance(m, nn.Linear):
            if std is not None:
                nn.init.normal_(m.weight, std=std)
            else:
                # print("[INFO] Using Xavier initialization for MLP.")
                gain = nn.init.calculate_gain(nonlinearity='relu')
                nn.init.xavier_normal_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)


class DeformationMLP(nn.Module):
    def __init__(
        self, 
        n_layer,
        layer_size,
        input_dim,
        output_dim,
        additional_input_dim=0,
        data_input_range_min=None,
        data_input_range_max=None,
        mlp_input_range_min=-1.,
        mlp_input_range_max=1.,
        output_range_min=-1.,
        output_range_max=1.,
        non_linearity=nn.ReLU(),
        final_non_linearity=None,
        positional_encoding=None,
        frequency_pos_encoding_freqs=4,
        ):
        """_summary_

        Args:
            n_layer (_type_): _description_
            layer_size (_type_): _description_
            input_dim (_type_): _description_
            output_dim (_type_): _description_
            additional_input_dim (int, optional): _description_. Defaults to 0.
            data_input_range_min (_type_, optional): _description_. Defaults to None.
            data_input_range_max (_type_, optional): _description_. Defaults to None.
            mlp_input_range_min (_type_, optional): _description_. Defaults to -1..
            mlp_input_range_max (_type_, optional): _description_. Defaults to 1..
            output_range_min (_type_, optional): _description_. Defaults to -1..
            output_range_max (_type_, optional): _description_. Defaults to 1..
            non_linearity (_type_, optional): Nonlinearity to use in the MLP. Defaults to nn.ReLU().
            final_non_linearity (_type_, optional): If None, no nonlinearity is applied after the last layer. Defaults to None.
            positional_encoding (_type_, optional): Positional encoding to use on the spatial input. 
                If None, no positional encoding is used. Defaults to None.
            frequency_pos_encoding_freqs (int, optional): _description_. Defaults to 4.

        Raises:
            ValueError: _description_
        """
        super(DeformationMLP, self).__init__()
        
        self.n_layer = n_layer
        self.layer_size = layer_size
        self.non_linearity = non_linearity
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.additional_input_dim = additional_input_dim
        
        self.data_input_range_min = data_input_range_min
        self.data_input_range_max = data_input_range_max

        self.mlp_input_range_min = mlp_input_range_min
        self.mlp_input_range_max = mlp_input_range_max

        self.output_range_min = output_range_min
        self.output_range_max = output_range_max
        
        self.final_non_linearity = final_non_linearity
        
        self._positional_encoding = positional_encoding
        self.use_positional_encoding = positional_encoding is not None
        self.frequency_pos_encoding_freqs = frequency_pos_encoding_freqs
        
        # Positional encoding
        if positional_encoding == 'frequency':
            self.positional_encoding = FrequencyPositionalEncoding(input_dim, frequency_pos_encoding_freqs)
            first_layer_input_dim = additional_input_dim + input_dim * 2 * frequency_pos_encoding_freqs
        elif positional_encoding is None:
            first_layer_input_dim = additional_input_dim + input_dim
            print("No positional encoding.")
        else:
            raise ValueError("Unknown positional encoding.")
        
        # MLP layers
        layers = nn.ModuleList()
        layers.append(nn.Linear(first_layer_input_dim, layer_size))
        layers.append(non_linearity)
        for i in range(n_layer-2):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(non_linearity)
        layers.append(nn.Linear(layer_size, output_dim))
        if final_non_linearity is not None:
            layers.append(final_non_linearity)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, additional_input=None):
        if (additional_input is None) and self.additional_input_dim > 0:
            raise ValueError("Additional input is required.")
        
        # Rescale input if needed
        if self.data_input_range_min is not None and self.data_input_range_max is not None:
            # --Bad rescaling
            # res = (x - self.data_input_range_min) / (self.data_input_range_max - self.data_input_range_min)
            # res = res * (self.mlp_input_range_max - self.mlp_input_range_min) + self.mlp_input_range_min
            
            # --Good rescaling
            x_center = (self.data_input_range_max + self.data_input_range_min) / 2
            x_scale = (self.data_input_range_max - self.data_input_range_min) / 2
            res = (x - x_center) / x_scale

            input_center = (self.mlp_input_range_max + self.mlp_input_range_min) / 2
            input_scale = (self.mlp_input_range_max - self.mlp_input_range_min) / 2
            res = res * input_scale + input_center
        else:
            res = x
            
        # Apply positional encoding
        if self.use_positional_encoding:
            res = self.positional_encoding(res)
        
        # Concatenate additional input
        if additional_input is not None:
            res = torch.cat([res, additional_input], dim=-1)
        
        # Apply MLP
        res = self.mlp(res)
        
        # Rescale output if needed
        if self.output_range_min is not None and self.output_range_max is not None:
            # --Bad rescaling
            # res = res * (self.output_range_max - self.output_range_min) + self.output_range_min
            
            # --Good rescaling
            output_center = (self.output_range_max + self.output_range_min) / 2
            output_scale = (self.output_range_max - self.output_range_min) / 2
            res = res * output_scale + output_center
        
        return res
