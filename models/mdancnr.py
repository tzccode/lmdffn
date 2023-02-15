import torch
import torch.nn as nn
from models.common import MeanShift
import torch.nn.functional as F
from thop import profile
from models.deformable_conv import ConvOffset2D
from torchstat import stat


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return y


class SpatialAttention(nn.Module):
    def __init__(self, n_feats=64):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, groups=n_feats)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        output = self.sigmoid(x)
        return output


class ReConstructAttention(nn.Module):
    def __init__(self, n_feats=64):
        super(ReConstructAttention, self).__init__()
        self.att = SpatialAttention(n_feats)

    def forward(self, x):
        att_weight = self.att(x)
        return x * att_weight


class CBAM(nn.Module):
    def __init__(self, channel=64):
        super(CBAM, self).__init__()
        self.channel_attention = CCALayer(channel=channel)
        self.spatial_attention = SpatialAttention(n_feats=channel)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class LightAC(nn.Module):
    def __init__(self, n_feat=64):
        super(LightAC, self).__init__()
        per_channel = n_feat // 4
        self.per_channel = per_channel
        self.conv_1_3 = nn.Conv2d(per_channel, per_channel, kernel_size=(1, 3), padding=(0, 1))

        self.conv_3_3 = nn.Sequential(
            nn.Conv2d(per_channel, per_channel * 4, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(per_channel * 4, per_channel, kernel_size=3, padding=1)
        )

        # self.conv_3_3 = nn.Conv2d(per_channel, per_channel, kernel_size=3, padding=1)

        self.conv_3_1 = nn.Conv2d(per_channel, per_channel, kernel_size=(3, 1), padding=(1, 0))

        self.act = nn.ReLU(inplace=True)
        self.conv_concat = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0)
        self.cbam = CBAM(channel=n_feat)

        self.scale_res = Scale(1.0)
        self.scale_identity = Scale(1.0)

    def forward(self, input):
        input_1_3, input_3_3, input_3_1, input_identity = torch.split(input, (self.per_channel, self.per_channel, self.per_channel, self.per_channel), dim=1)

        feat_1_3 = self.conv_1_3(input_1_3)
        feat_3_3 = self.conv_3_3(input_3_3)
        feat_3_1 = self.conv_3_1(input_3_1)

        concat_feat = torch.cat([feat_1_3, feat_3_3, feat_3_1, input_identity], dim=1)
        concat_feat = self.act(concat_feat)
        concat_feat = channel_shuffle(concat_feat, groups=4)
        concat_feat = self.cbam(concat_feat)
        output = self.conv_concat(concat_feat)
        return self.scale_res(output) + self.scale_identity(input)


class ReconstructModule(nn.Module):
    def __init__(self, scale, n_feats=64, n_colors=3):
        super(ReconstructModule, self).__init__()
        out_feats = scale*scale*n_colors
        self.tail_k3 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=3//2, dilation=1, groups=n_feats),
            nn.Conv2d(n_feats, out_feats, 1, padding=0)
        )
        self.tail_k5 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 5, padding=5//2, dilation=1, groups=n_feats),
            nn.Conv2d(n_feats, out_feats, 1, padding=0)
        )
        self.pixelshuffle = nn.PixelShuffle(scale)
        self.scale_k3 = Scale(0.5)
        self.scale_k5 = Scale(0.5)

    def forward(self, x):
        x0 = self.pixelshuffle(self.scale_k3(self.tail_k3(x)))
        x1 = self.pixelshuffle(self.scale_k5(self.tail_k5(x)))

        return x0+x1


class MDANCNR(nn.Module):
    def __init__(self, scale, in_channels=3, n_feat=64, n_light_acs=10, rgb_range=255.):
        super(MDANCNR, self).__init__()

        self.rgb_range = rgb_range
        self.sub_mean = MeanShift(self.rgb_range)
        self.add_mean = MeanShift(self.rgb_range, sign=1)

        # head module
        head = nn.Conv2d(in_channels, n_feat, 3, 1, 1)

        # body module
        body = nn.ModuleList()
        for _ in range(n_light_acs):
            body.append(LightAC(n_feat=n_feat))

        self.conv_concat = nn.Sequential(
            nn.Conv2d(n_feat * n_light_acs, n_feat, kernel_size=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)
        )

        # tail module
        tail = ReconstructModule(scale, n_feat, n_colors=in_channels)
        out_feats = scale * scale * in_channels

        # skip module

        skip = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=3 // 2, groups=in_channels),
            nn.Conv2d(in_channels, out_feats, kernel_size=1),
            nn.PixelShuffle(scale)
        )

        self.head = head
        self.body = body
        self.tail = tail
        self.skip = skip

    def forward(self, x):
        x = self.sub_mean(x)
        s = self.skip(x)
        x = self.head(x)
        feat_concat = []
        for layer in self.body:
            y = layer(x)
            feat_concat.append(y)
            x = y
        feat_concat = torch.cat(feat_concat, dim=1)
        x = self.conv_concat(feat_concat)
        x = self.tail(x)
        x = x + s
        sr = self.add_mean(x)
        return sr

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0 or  name.find('skip') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


if __name__ == "__main__":
    model = MDANCNR(scale=2)
    # model = AcResNet_Thin_Random(scale=2)
    input = torch.randn(1, 3, 360, 240)
    flops, params = profile(model, (input,))
    print('flops: ', flops, 'params: ', params)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    stat(model, (3, 360, 240))
