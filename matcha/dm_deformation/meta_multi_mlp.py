import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from .encodings import FrequencyPositionalEncoding


def initialize_meta_multi_mlp_weights(network, std=None):
    if std is not None:
        print("[INFO] Using normal initialization for MLP with std={}".format(std))
    for m in network.modules():
        if isinstance(m, MetaMultiLinear):
            for i in range(m.n_heads):
                if std is not None:
                    nn.init.normal_(m.cond_weight[i], std=std)
                else:
                    # print("[INFO] Using Xavier initialization for MLP.")
                    gain = nn.init.calculate_gain(nonlinearity='relu')
                    nn.init.xavier_normal_(m.cond_weight[i], gain=gain)
            nn.init.zeros_(m.cond_bias)
            
        if isinstance(m, nn.Linear):
            if std is not None:
                nn.init.normal_(m.weight, std=std)
            else:
                # print("[INFO] Using Xavier initialization for MLP.")
                gain = nn.init.calculate_gain(nonlinearity='relu')
                nn.init.xavier_normal_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)


class MetaMultiLinear(nn.Module):
    r"""Applies N linear transformations to N batches of incoming data: :math:`y = xA^T + b`

    Args:
        n_heads: number of heads
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(n_heads, *, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(n_heads, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Examples::
        >>> m = MultiLinear(10, 32, 64)
        >>> x = torch.rand(10, 10000, 32)
        >>> y = m(x)
        >>> print(y.size())
        torch.Size([10, 10000, 64])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    cond_weight: torch.Tensor

    def __init__(
        self, 
        n_heads: int,
        in_features: int, 
        cond_in_features: int,
        out_features: int, 
        bias: bool = True,
        device=None, 
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.cond_in_features = cond_in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.cond_weight = nn.Parameter(torch.empty((n_heads, out_features * (in_features+1), cond_in_features), **factory_kwargs))
        if bias:
            self.cond_bias = nn.Parameter(torch.empty((n_heads, out_features * (in_features+1)), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('cond_bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.cond_weight, a=math.sqrt(5))
        if self.cond_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.cond_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.cond_bias, -bound, bound)

    def forward(self, input: torch.Tensor, cond:torch.Tensor) -> torch.Tensor:
        # Get conditional weights for MultiLinear, conditioned on cond
        weight = torch.bmm(
            cond,  # (n_heads, batch_size, cond_in_features)
            self.cond_weight.transpose(1, 2)  # (n_heads, out_features*(in_features+1), cond_in_features)
        )  # (n_heads, batch_size, out_features*(in_features+1))
        if self.cond_bias is not None:
            weight = weight + self.cond_bias[:, None, :]
        weight = weight.reshape(self.n_heads, -1, self.out_features, self.in_features + 1)  # (n_heads, batch_size, out_features, in_features + 1)

        # Apply MultiLinear transformation
        output = torch.bmm(
            nn.functional.pad(
                input=input.reshape(-1, 1, self.in_features), # (n_heads*batch_size, 1, in_features)
                pad=(0, 1), 
                value=1.,  # Add bias
            ),  # (n_heads*batch_size, 1, in_features + 1)
            weight.reshape(-1, self.out_features, self.in_features + 1).transpose(1, 2)  # (n_heads*batch_size, in_features + 1, out_features)
        ).view(self.n_heads, -1, self.out_features)  # (n_heads, batch_size, out_features)
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, cond_in_features={}, cond_bias={}'.format(
            self.in_features, self.out_features, self.cond_in_features, self.cond_bias is not None
        )
        
        
class DeformationMetaMultiMLP(nn.Module):
    def __init__(
        self, 
        n_heads,
        n_layer,
        layer_size,
        input_dim,
        cond_dim,
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
        super(DeformationMetaMultiMLP, self).__init__()
        
        self.n_heads = n_heads
        self.n_layer = n_layer
        self.layer_size = layer_size
        self.non_linearity = non_linearity
        
        self.input_dim = input_dim
        self.cond_dim = cond_dim
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
        layers.append(MetaMultiLinear(
            n_heads=n_heads,
            in_features=first_layer_input_dim, 
            cond_in_features=cond_dim,
            out_features=layer_size, 
        ))
        layers.append(non_linearity)
        for i in range(n_layer-2):
            layers.append(MetaMultiLinear(
                n_heads=n_heads, 
                in_features=layer_size, 
                cond_in_features=cond_dim,
                out_features=layer_size
            ))
            layers.append(non_linearity)
        layers.append(MetaMultiLinear(
            n_heads=n_heads, 
            in_features=layer_size, 
            cond_in_features=cond_dim,
            out_features=output_dim
        ))
        if final_non_linearity is not None:
            layers.append(final_non_linearity)
        self.mlp = layers

    def forward(self, x, cond, additional_input=None):
        # x should have shape (batch_size, input_dim)
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
        # res, _ = self.mlp(res, cond)
        for layer in self.mlp:
            if isinstance(layer, MetaMultiLinear):
                res = layer(res, cond)
            else:
                res = layer(res)
        
        # Rescale output if needed
        if self.output_range_min is not None and self.output_range_max is not None:
            # --Bad rescaling
            # res = res * (self.output_range_max - self.output_range_min) + self.output_range_min
            
            # --Good rescaling
            output_center = (self.output_range_max + self.output_range_min) / 2
            output_scale = (self.output_range_max - self.output_range_min) / 2
            res = res * output_scale + output_center
        
        return res
