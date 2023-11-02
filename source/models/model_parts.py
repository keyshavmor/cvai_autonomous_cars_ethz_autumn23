import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet


class BasicBlockWithDilation(torch.nn.Module):
    """Workaround for prohibited dilation in BasicBlock in 0.4.0"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockWithDilation, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = resnet.conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = resnet.conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


_basic_block_layers = {
    'resnet18': (2, 2, 2, 2),
    'resnet34': (3, 4, 6, 3),
}


def get_encoder_channel_counts(encoder_name):
    is_basic_block = encoder_name in _basic_block_layers
    ch_out_encoder_bottleneck = 512 if is_basic_block else 2048
    ch_out_encoder_4x = 64 if is_basic_block else 256
    return ch_out_encoder_bottleneck, ch_out_encoder_4x


class Encoder(torch.nn.Module):
    def __init__(self, name, **encoder_kwargs):
        super().__init__()
        encoder = self._create(name, **encoder_kwargs)
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **encoder_kwargs):
        if name not in _basic_block_layers.keys():
            fn_name = getattr(resnet, name)
            model = fn_name(**encoder_kwargs)
        else:
            # special case due to prohibited dilation in the original BasicBlock
            pretrained = encoder_kwargs.pop('pretrained', False)
            progress = encoder_kwargs.pop('progress', True)
            model = resnet._resnet(
                name, BasicBlockWithDilation, _basic_block_layers[name], pretrained, progress, **encoder_kwargs
            )

        replace_stride_with_dilation = encoder_kwargs.get('replace_stride_with_dilation', (False, False, False))
        assert len(replace_stride_with_dilation) == 3
        if replace_stride_with_dilation[0]:
            model.layer2[0].conv2.padding = (2, 2)
            model.layer2[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[1]:
            model.layer3[0].conv2.padding = (2, 2)
            model.layer3[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[2]:
            model.layer4[0].conv2.padding = (2, 2)
            model.layer4[0].conv2.dilation = (2, 2)
        return model

    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, x):
        """
        DeepLabV3+ style encoder
        :param x: RGB input of reference scale (1x)
        :return: dict(int->Tensor) feature pyramid mapping downscale factor to a tensor of features
        """
        out = {1: x}
        sz_in = x.shape[3]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.maxpool(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer1(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer2(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer3(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer4(x)
        self.update_skip_dict(out, x, sz_in)

        return out


class DecoderDeeplabV3p(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
        super(DecoderDeeplabV3p, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following
        self.project_skip = torch.nn.Sequential(
            torch.nn.Conv2d(skip_4x_ch, 48, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
        )

        self.features_to_predictions = torch.nn.Sequential(
            torch.nn.Conv2d(bottleneck_ch + 48, 256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, num_out_ch, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.

        features_skip_4x = self.project_skip(features_skip_4x)

        features_bottleneck_4x = F.interpolate(
            features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        )

        concat_features = torch.cat([features_bottleneck_4x, features_skip_4x], dim=1)
        predictions_4x = self.features_to_predictions(concat_features)

        return predictions_4x, features_bottleneck_4x


class ASPPpart(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )


class ASPP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rates=(3, 6, 9)):
        super().__init__()
        # TODO: Implement ASPP properly instead of the following
        
        # 1*1 conv
        self.aspp1 = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        # 3*3 astrous conv with rates (3, 6, 9)
        self.aspp3_3 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[0], dilation=rates[0])
        self.aspp3_6 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[1], dilation=rates[1])
        self.aspp3_9 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[2], dilation=rates[2])
        # Pooling
        self.aspp_pooling = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels, out_channels, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels), # CHECK
            torch.nn.ReLU()
        )

        # 1*1 conv after concat
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )

    def forward(self, x):
        # TODO: Implement ASPP properly instead of the following
        x1 = self.aspp1(x)
        x2 = self.aspp3_3(x)
        x3 = self.aspp3_6(x)
        x4 = self.aspp3_9(x)
        x5 = self.aspp_pooling(x)
        x5 = F.interpolate(x5, size=x.shape[-2:], mode='bilinear', align_corners=False)

        concat_x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.conv_out(concat_x)

        return out


class MLP(torch.nn.Module):
    def __init__(self, dim, expansion):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj1 = nn.Linear(dim, int(dim * expansion))
        self.act = nn.GELU()
        self.proj2 = nn.Linear(int(dim * expansion), dim)

    def forward(self, x):
        """
        MLP with pre-normalization with one hidden layer
        :param x: batched tokens with dimesion dim (B,N,C)
        :return: tensor (B,N,C)
        """
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        return x


class SelfAttention(torch.nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.out = nn.Linear(dim, dim)
        self.temperature = 1.0
        self.dropout = nn.Dropout(dropout)

        # TODO: Implement self attention, you need a projection
        # and the normalization layer
        
        self.head_dim = dim // self.num_heads
        assert self.head_dim * self.num_heads == dim, "You should set embed_dim so that it can be devided by num_heads"
        
        self.layer_norm = nn.LayerNorm(dim)
        
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

        
    def forward(self, x, pos_embed):
        """
        Pre-normalziation style self-attetion
        :param x: batched tokens with dimesion dim (B,N,C)
        :param pos_embed: batched positional with shape (B, N, C)
        :return: tensor processed with output shape same as input
        """
        B, N, C = x.shape

        # TODO: Implement self attention, you need:
        # 1. obtain query, key, value
        # 2. if positional embed is not none add to the query tensor
        # 3. compute the similairty between query and key (dot product)
        # 4. obtain the convex combination of the values based on softmax of similarity
        # Remember that the num_heads speciifc the number of indepdendt heads to unpack
        # the operation in. Namely 2 heads, means that the channels are unpacked into 
        # 2 independent: something.reshape(B, N, self.num_heads, C // self.num_heads)
        # and rearrange accordingly to perform the operation such that you work only with
        # N and self.dim // self.num_heads
        # Remember that also the positional embedding needs to follow a similar rearrangement
        # to be consistent with the shapes of the query
        # Remember to rearrange the output tensor such that the output shape is B N C again
        
        DEBUG = False
        x = x.transpose(1, 2) # (B, C, N)
        
        q = self.proj_q(x) # (B, C, N)
        k = self.proj_k(x) # (B, C, N)
        v = self.proj_v(x) # (B, C, N)
        
        if pos_embed:
            q += pos_embed
            
        x = x.transpose(1, 2) # (B, N, C)
        
        q = q.reshape(B, self.num_heads, self.head_dim, C) # (B, num_heads, head_dim, C)
        k = k.reshape(B, self.num_heads, self.head_dim, C) # (B, num_heads, head_dim, C)
        v = v.reshape(B, self.num_heads, self.head_dim, C) # (B, num_heads, head_dim, C)
        
        dim_k = q.shape[2] # = head_dims
        multi_attn_weight = torch.matmul(q, k.transpose(2, 3)) # [B, num_heads, head_dim, C] * [B, num_heads, C, head_dim] = [B, num_heads, head_dim, head_dim]
        if DEBUG:
            print("multi_attn_weight", multi_attn_weight.shape, B, N, C, self.num_heads, self.head_dim)
        
        multi_attn_weight = multi_attn_weight / dim_k
        multi_attn_weight = F.softmax(multi_attn_weight, dim=3) # [B, num_heads, head_dim]
        if DEBUG:
            print("after softmax", multi_attn_weight.shape, B, N, C, self.num_heads, self.head_dim)
        
        multi_attn_out = torch.matmul(multi_attn_weight, v) # [B, num_heads, head_dim, C] * [B, num_heads, head_dim] = [B, num_heads, head_dim, C]
        if DEBUG:
            print("multi_attn_out", multi_attn_out.shape, B, N, C, self.num_heads, self.head_dim)
        attn_out = multi_attn_out.reshape(B, N, C)
        if DEBUG:
            print("attn_out", attn_out.shape, B, N, C, self.num_heads, self.head_dim)
            
        attn_out = attn_out.transpose(1, 2) # (B, N, C)
        o = self.out(attn_out)
        o = o.transpose(1, 2)

        return o


class TransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_heads=4, expansion=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = dim
        self.mlp = MLP(dim, expansion=expansion)
        self.attn = SelfAttention(dim=dim, num_heads=num_heads)

    def forward(self, x, pos_embed=None):
        x = self.attn(x, pos_embed) + x
        x = self.mlp(x) + x
        return x
    

class LatentsExtractor(torch.nn.Module):
    def __init__(self, num_latents: int = 256):
        super().__init__()
        edge = num_latents ** 0.5
        assert edge.is_integer(), "Remeber to give num_bins a number whose sqrt is still an integere, e.g., 256"
        self.edge = int(edge)
    
    def forward(self, x):
        B, C = x.shape[:2]
        x_down = F.interpolate(x, size=(self.edge, self.edge), mode="bilinear", align_corners=False)
        x_down_flat = x_down.permute(0, 2, 3, 1).view(B, self.edge*self.edge, C)
        return x_down_flat
    
