from tokenize import group
import torch as th
from torch._C import Size
import torch.nn as nn

import torch.nn.functional as F
from torchdiffeq import odeint
import torchdiffeq

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])

    # return neles / 10**6 if Mb else neles
    return neles


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(dim, 1))
            self.gamma = nn.Parameter(th.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        print(x)
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D conv transpose in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                in_channels=256,
                conv_channels=512,
                groups=16,
                Sc = 512,
                kernel_size=3,
                dilation=1,
                norm="cLN",
                causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv nxBxT -> nXHXT
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # shuflling 
        
        groups= (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation* (kernel_size - 1))
        # shuflling 
        #outchannl=(conv_channels//groups)
        # shuflled conv
        groupoutchnl=256
        dcon_group_output= groupoutchnl+groupoutchnl
        # shuflled conv
        self.shuffgroupconv=nn.Conv1d(
            conv_channels,
            groupoutchnl,
            kernel_size,
            groups=groups,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        # depthwise conv 
        self.tanh1 = nn.Tanh()
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.shuffgroup=ChannelShuffle(conv_channels,groups)
        self.conv1x1_2 = Conv1D(conv_channels, groupoutchnl, 1)    
        self.sigmoid1 = nn.Sigmoid()   
        #self.conv1x1_2 = Conv1D(dcon_group_output, dcon_group_output, 1)    
        self.prelu2 = nn.PReLU()
        self.lnorm2 = build_norm(norm, groupoutchnl*2)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(groupoutchnl*2, in_channels, 1, bias=True)
        #1x1 conv skip-connection
        self.skip_out = nn.Conv1d(groupoutchnl*2, Sc, 1, bias=True)
        self.causal = causal
        self.dconv_pad = dconv_pad






    def forward(self, x):
        y = self.conv1x1(x)
        
        if __name__ == "__main__":
            print('1D blick after fist 1x1Conv size', y.size())
            
        y = self.lnorm1(self.prelu1(y))
        sh=self.shuffgroup(y)
        shuffcov=self.shuffgroupconv(sh)
        shufftan=self.tanh1(shuffcov)
        shuffsigm=self.sigmoid1(shuffcov)       
        print("shape after shuffle",sh.size())
        print("shape of shuffle groupe conv",shuffcov.size())
        
        y = self.dconv(y)
        y=self.conv1x1_2(y)
        depsigm=self.sigmoid1(y)
        deptan=self.tanh1(y)
        _x_up=shufftan*depsigm
        _x_down=shuffsigm*deptan
        depsigm=self.sigmoid1(y)
        y = th.cat(( _x_up, _x_down), axis=1)
        #y=y+shuffcov
        #x_out = self.Multiply()([x_sigmoid, x_tanh])
        # y = th.cat(( y, shuffcov), axis=1)

        print("shape before 1x1",  y.size())
        # y=self.conv1x1_2(y)       
        # newArray= self.conv1x1(newArray)
        print("shape after 1x1",  y.size())
        if __name__ == "__main__":
            print('1D Conv block after group cov size', y.size())
        
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        out = self.sconv(y)
        skip = self.skip_out(y)
        x = x + out

        return skip, x
# def channel_shuffle(x,
#                     groups):
#     """
#     Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
#     https://arxiv.org/abs/1707.01083.
#     Parameters:
#     ----------
#     x : Tensor
#         Input tensor.
#     groups : int
#         Number of groups.
#     Returns
#     -------
#     Tensor
#         Resulted tensor.
#     """
#     batch, channels, height = x.size()
#     # assert (channels % groups == 0)
#     channels_per_group = channels // groups
#     x = x.view(batch, groups, channels_per_group, height)
#     x = th.transpose(x, 1, 2).contiguous()
#     x = x.view(batch, channels, height)
#     return x



class ChannelShuffle(nn.Module):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """


    
    def __init__(self,
                channels,
                groups):
        super(ChannelShuffle, self).__init__()
        # assert (channels % groups == 0)
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groups

    def forward(self, x):
        return channel_shuffleforsound(x, self.groups)





def channel_shuffleforsound(x,
                    groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083. The alternative version.
    Parameters:
    ----------
    x : Tensor
        Input tensor.
    groups : int
        Number of groups.
    Returns
    -------
    Tensor
        Resulted tensor.
    """
    batch, inchannel, height =x.size()
    channels_per_group = inchannel // groups
    
    print("size of x",x.size())
    # assert (channels % groups == 0)
    
    x = x.view(batch, channels_per_group, groups, height)
    x = th.transpose(x, 1, 2).contiguous()
    x = x.view(batch, inchannel, height)
    return x



def channel_shuffle2(x, groups):
    """
    Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'
    https://arxiv.org/abs/1707.01083. The alternative version.
    Parameters:
    ----------
    x : Tensor
        Input tensor.
    groups : int
        Number of groups.
    Returns
    -------
    Tensor
        Resulted tensor.
    """
    batch, channels, height = x.size()
    # assert (channels % groups == 0)
    channels_per_group = channels // groups
    x = x.view(batch, channels_per_group, groups, height)
    x = th.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height)
    return x


class ChannelShuffle2(nn.Module):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    The alternative version.
    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self,
                channels,
                groups):
        super(ChannelShuffle2, self).__init__()
        # assert (channels % groups == 0)
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groups

    def forward(self, x):
        return channel_shuffle2(x, self.groups)

# -----------------node--------------------------------------------
class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, t, x):
        return self.net(x)

class ODEBlock(nn.Module):
    def __init__(self, odefunc, method='rk4', rtol=1e-3, atol=1e-4):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(self, x):
        t = th.tensor([0, 1]).float().to(x.device)
        out = torchdiffeq.odeint(self.odefunc, x, t, method=self.method, rtol=self.rtol, atol=self.atol)
        return out[1]

#-------------------------------------node--------------------------------



#  *******************************************************  

# -----------------FusionModel------------------------
class MS_SL2_split_model(nn.Module):
    def __init__(self,
                 L=16,
                 N=512,
                 X=8,
                 R=1,
                 B=256,
                 Sc=256,
                 Slice=1,
                 H=512,
                 P=3,
                 norm="gLN",
                 num_spks=2,
                 non_linear="sigmoid",
                 causal=False):
        super(MS_SL2_split_model, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "softmax": F.softmax
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}".format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        self.encoder_1d = Conv1D(1, N, L, stride=L // 2, padding=0)
        self.ln = ChannelWiseLayerNorm(N)
        self.proj = Conv1D(N, B, 1)
        self.slices = self._build_slices(
            Slice,
            R,
            X,
            Sc=Sc,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)
        max = 0.07331312
        min = -0.0814228
        self.wList = nn.Parameter((max - min) * th.rand(4) + min, requires_grad=True)
        self.PRelu = nn.PReLU()
        self.mask = Conv1D(Sc, num_spks * N, 1)
        self.decoder_1d = ConvTrans1D(N, 1, kernel_size=L, stride=L // 2, bias=True)
        self.num_spks = num_spks
        self.R = R
        self.X = X
        self.slice = Slice
        self.ode_block = ODEBlock(ODEFunc(N, H))  # Adjusted input_dim to N and hidden_dim to H

    def _build_blocks(self, num_blocks, **block_kwargs):
        blocks = [
            Conv1DBlock(**block_kwargs, dilation=(2 ** b))
            for b in range(num_blocks)
        ]
        return nn.Sequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        repeats = [
            self._build_blocks(num_blocks, **block_kwargs)
            for r in range(num_repeats)
        ]
        return nn.Sequential(*repeats)

    def _build_slices(self, num_slice, num_repeats, num_blocks, **block_kwargs):
        slices = [
            self._build_repeats(num_repeats, num_blocks, **block_kwargs)
            for r in range(num_slice)
        ]
        return nn.Sequential(*slices)

    def forward(self, x):
        if x.dim() >= 3:
            raise RuntimeError("{} accept 1/2D tensor as input, but got {:d}".format(self.__name__, x.dim()))
        if x.dim() == 1:
            x = th.unsqueeze(x, 0)
        w = F.relu(self.encoder_1d(x))
        y = self.proj(self.ln(w))
        th.skip_into_weight = 0
        th.skip_into_weight_sum = 0
        skip_connection = 0
        Slice_input = y
        Tcn_into_weight = []
        Tcn_output_result = 0
        for Slice in range(self.slice):
            if __name__ == "__main__":
                print('slice input size', y.size())
            for i in range(self.R):
                for j in range(self.X):
                    if __name__ == "__main__":
                        print('1D Conv block input size', y.size())
                    skip, y = self.slices[Slice][i][j](y)
                    skip_connection = skip_connection + skip
                    if __name__ == "__main__":
                        print('finished 1D Conv block skip_connection size', skip.size())
                        print('finished 1D Conv block ouput size', y.size())
                        print("Weight values", self.wList)
            if Slice == 0:
                slic1_into_weight = th.einsum('i,jkl->ijkl', [self.wList, skip_connection])
                Slices_Output = th.einsum('ijkl->jkl', [slic1_into_weight])
                print("Slice 1 output dim", Slices_Output.dim())
            elif Slice == 1:
                slic2_into_weight = th.einsum('i,jkl->ijkl', [self.wList, skip_connection])
                Slice2_row_sum = th.einsum('ijkl->jkl', [slic2_into_weight])
                print("Slice number", Slice)
                Slices_Output = th.add(Slices_Output, Slice2_row_sum)
                print("Slice 2 output dim", Slices_Output.dim())
            else:
                print("Slice number", Slice)
                slic3_into_weight = th.einsum('i,jkl->ijkl', [self.wList, skip_connection])
                Slice3_row_sum = th.einsum('ijkl->jkl', [slic3_into_weight])
                Slices_Output = th.add(Slices_Output, Slice3_row_sum)
                print("Slice 3 output dim", Slices_Output.dim())
            skip_connection = 0
            y = Slice_input
        y = self.PRelu(Slices_Output)
        e = th.chunk(self.mask(y), self.num_spks, 1)
        print("size of e", e[0].size())
        if __name__ == "__main__":
            print('after 1x1 Conv mask)', e[0].size())
        if self.non_linear_type == "softmax":
            m = self.non_linear(th.stack(e, dim=0), dim=0)
        else:
            m = self.non_linear(th.stack(e, dim=0))
        if __name__ == "__main__":
            print('after M Conv mask)', m.size())
        s = [w * m[n] for n in range(self.num_spks)]

        # Reshape for ODEBlock
        batch_size, num_channels, seq_len = w.size()
        s = [x.permute(0, 2, 1).contiguous().view(batch_size * seq_len, num_channels) for x in s]  # Fix reshape to match ODE block
        print("Shape of s after permute and view:", [x.shape for x in s])
        s = [self.ode_block(x) for x in s]
        s = [x.view(batch_size, seq_len, num_channels).permute(0, 2, 1) for x in s]
        print("Shape of s after ODEBlock and view:", [x.shape for x in s])

        for i, x in enumerate(s):
            print(f"Shape of s[{i}] after ODE block: {x.shape}")
        return [self.decoder_1d(x, squeeze=True) for x in s]

# Ensure to run the updated model
if __name__ == "__main__":
    x = th.rand(4, 1000)
    nnet = MS_SL2_split_model(norm="cLN", causal=False)
    print("ConvTasNet #param: {:.2f}".format(param(nnet)))
    x = nnet(x)
    print(nnet)
    s1 = x[0]
    print(s1.shape)
