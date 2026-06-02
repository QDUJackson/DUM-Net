# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair, _triple
import module.configs as configs
from torch.distributions.normal import Normal
from module.utils import *
from module.pdfNet import distNetwork
from module.SingleTransformer import SingleStageSwin
from module.EMA_Attention import EMA
from module.SoftplusSoftErf_Class import SoftplusSoftErf
logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        # num_heads = 8
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)

        # config.hidden_size = 128
        # self.num_attention_heads = 8

        # self.attention_head_size = 128 / 8 = 16

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # all_head_size 就是所有的头的size 就是hidden_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        # 输入节点数和输出节点数
        # WQ,WK,WV
        self.out = Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        # attention_dropout_rate 全都是0
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # hidden_states: [8,64,128]
        # [batch_size, patch_nums, sequence_length]

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # xWQ,xWK,xWV

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # query_layer shape: [8,8,64,16]
        # [batch_size, num_heads, patch_nums, sequence_length]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # QKT
        # [8,8,64,64]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 为了缩小规模
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)

        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        # mlp_dim = 3072
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        # 先变大再变小
        self.act_fn = ACT2FN["gelu"]

        self.dropout = Dropout(config.transformer["dropout_rate"])
        # 这里dropout给的是0.1
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        # 均匀初始化 确保在前向反向传播时信号的稳定
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        # 正态分布初始化，为了让偏执在初始时偏向于0

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size):
        super(Embeddings, self).__init__()
        # 这个地方输入的img_size很明显是一个三维图片的三个维度
        self.config = config

        down_factor = config.down_factor
        # down_factor = 2 Transformer之前的下采样次数等于之后的下采样次数

        patch_size = _pair(config.patches["size"])
        # 8 8这里是一个tuple

        n_patches = int(
            (img_size[0] / 2 ** down_factor // patch_size[0]) * (img_size[1] / 2 ** down_factor // patch_size[1]))
        # down_fator是下采样的次数，然后//是整除的意思，这样计算出patch的数量
        # 一共是 64个patch

        self.hybrid_model = CNNEncoder(config, n_channels=2)

        in_channels = config['encoder_channels'][-1]
        # 这里取的是32，也就是最底层的通道数
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # 这个地方的stride和kernel_size都是tuple表示在每个维度上的尺寸
        # 这里注意的是，每个patch最后变成了hidden_size个通道，也就是变成字条了
        # 用这种卷积直接切patch

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        # 要嵌入的位置信息
        self.dropout = Dropout(config.transformer["dropout_rate"])
        # 0.1 rate

    def forward(self, x):
        x, features = self.hybrid_model(x)

        # 这个时候还没切片
        x = self.patch_embeddings(x)  # (B, hidden_size, n_patches^(1/2), n_patches^(1/2) )

        # 到最下面之后切片
        x = x.flatten(2)

        # 将x从第3个维度开始的所有维度都展开，展平
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # 变成了 64个序列 hiddensize就是每个序列的长度
        # 一次只能交换两个维度 transpose
        embeddings = x + self.position_embeddings
        # 这个地方的broadcasting是因为每个batch同一位置的patch位置信息都应该是一样的

        embeddings = self.dropout(embeddings)

        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()

        self.hidden_size = config.hidden_size

        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)

        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)

        self.ffn = Mlp(config)

        self.attn = Attention(config, vis)

    def forward(self, x):
        # 整体的结构就是先Norm,然后MSA，然后再Norm，然后再MLP
        h = x

        x = self.attention_norm(x)
        x, weights = self.attn(x)

        x = x + h

        h = x
        # 准备跳接
        x = self.ffn_norm(x)
        x = self.ffn(x)

        x = x + h

        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()

        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        # 专门用于归一化序列化数据 其中有可学习的参数，基本就是均值为0，方差为1

        # 制造多层的Transformer
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            # 这个Block就是Transformer的块
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):

        attn_weights = []

        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)

            if self.vis:
                attn_weights.append(weights)

        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()

        # Transformer 部分分为 Embeddings 和 Encoder

        self.embeddings = Embeddings(config, img_size=img_size)
        # 在Embedding中就是CNNEncoder之后，然后切成patch 往后面Transformer里放就行了

        self.encoder = SingleStageSwin(
        input_resolution=(8, 8),
        dim=128,
        depth=10,
        num_heads=8,
        window_size=4  # 可以自己改成 8，看你想怎样分块
        )
        # 然后这里之后进行Transformer的12个layer

    def forward(self, input_ids):
        # CNN的Encoder是放在embedding中了
        # 而Transformer的各个层是放在encoder中了
        embedding_output, features = self.embeddings(input_ids)
        # 这些features得用到后面Decoder里 是倒叙后的Encoder信息
        # embedding_output是 [batch_size,n_patch,hidden_size]

        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)

        attn_weights = []


        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)


        if use_batchnorm:

            bn = nn.BatchNorm2d(out_channels)
            super(Conv2dReLU, self).__init__(conv, bn, relu)
        else:
            super(Conv2dReLU, self).__init__(conv, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # 保持角点，如果align_corners是True的话，可以保存角点，也就是原来的边界是保存的，这样对边界保持任务是友好的

        #self.ema = EMA(out_channels,None,8)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        #x = self.ema(x)

        return x


class DecoderCup(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.config = config

        self.down_factor = config.down_factor
        # down_factor = 2
        head_channels = config.conv_first_channel
        # 256
        self.img_size = img_size

        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        # config.decoder_channels = (96, 48, 32, 32, 16)

        in_channels = [head_channels] + list(decoder_channels[:-1])
        # [512,96,48,32,32,16]
        out_channels = decoder_channels

        self.patch_size = _triple(config.patches["size"])
        # (8,8,8)
        skip_channels = self.config.skip_channels
        # config.skip_channels = (32, 32, 32, 32, 16)

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):

        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)

        h, w = (self.img_size[0] // 2 ** self.down_factor // self.patch_size[0]), (
                    self.img_size[1] // 2 ** self.down_factor // self.patch_size[1])
        # 经检验得 l,h,w确实是原图大小的1/32倍，但是他在Decoder时，被用了up

        x = hidden_states.permute(0, 2, 1)

        x = x.contiguous().view(B, hidden, h, w)

        x = self.conv_more(x)


        for i, decoder_block in enumerate(self.blocks):

            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None

            x = decoder_block(x, skip=skip)
        return x


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer

    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        '''
        下面我们需要调整一下bn 和 relu的顺序
        '''
        #nn.GroupNorm(num_groups=4, num_channels=mid_channels),
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        #self.ema = EMA(out_channels)

    def forward(self, x):

        x_double = self.double_conv(x)
        #x_ema = self.ema(x_double)

        return x_double


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class CNNEncoder(nn.Module):
    def __init__(self, config, n_channels=2):
        super(CNNEncoder, self).__init__()

        self.n_channels = n_channels
        decoder_channels = config.decoder_channels

        encoder_channels = config.encoder_channels
        # config.encoder_channels = (16, 32, 32)

        self.down_num = config.down_num
        # down_num = 2 这个是需要额外做的down数

        self.inc = DoubleConv(n_channels, encoder_channels[0])

        self.down1 = Down(encoder_channels[0], encoder_channels[1])
        self.down2 = Down(encoder_channels[1], encoder_channels[2])

        self.width = encoder_channels[-1]

    def forward(self, x):
        features = []
        x1 = self.inc(x)

        # 先变成16个通道
        features.append(x1)

        x2 = self.down1(x1)
        # 先最大池化下采样，然后用两次conv，配合两次ReLU

        features.append(x2)

        feats = self.down2(x2)



        # 经过两次下采样到了 Transformer的部分
        # 再下采样，同理

        features.append(feats)

        feats_down = feats

        for i in range(self.down_num):
            feats_down = nn.MaxPool2d(2)(feats_down)
            #print(feats_down.shape)
            features.append(feats_down)

        # 这个features[::-1]的表达，意思是让features倒序返回
        # 这个feats就是进Transformer 12layer之前的数据
        return feats, features[::-1]


class RegistrationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        conv2d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv2d.weight.shape))
        conv2d.bias = nn.Parameter(torch.zeros(conv2d.bias.shape))

        super().__init__(conv2d)


class ViTVNet(nn.Module):
    def __init__(self, config, img_size=(256, 256), int_steps=7, vis=False):
        super(ViTVNet, self).__init__()

        self.transformer = Transformer(config, img_size, vis)

        self.decoder = DecoderCup(config, img_size)

        # Transformer里有encoder
        self.reg_head = RegistrationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config['n_dims'],
            kernel_size=3,
        )
        # 这个像最后生成变形场的卷积
        self.spatial_trans = SpatialTransformer(img_size)
        self.config = config
        # self.integrate = VecInt(img_size, int_steps)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Get_Histogram = distNetwork(0.01, device)

    def forward(self, x):
        # 整个网络框架分为三部分 Transformer decoder reg_head
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)

        x = self.decoder(x, features)

        flow = self.reg_head(x)

        return flow


CONFIGS = {
    'ViT-V-Net': configs.get_2DReg_config(),
}


def test_ViTVNet():
    # 假设 img_size 是 (64, 256, 256)，batch size 为 1
    config = CONFIGS['ViT-V-Net']
    img_size = (256, 256)
    model = ViTVNet(config, img_size=img_size)

    # 随机生成一个输入张量，假设输入为 2 通道（比如用于配准的两幅图像）
    x = torch.randn(8, 2, *img_size)  # shape: (batch, channels, depth, height, width)

    # 执行前向传播并检查每个阶段的输出尺寸

    flow = model(x)

    return flow


test_ViTVNet()
