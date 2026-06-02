import torch
import torch.nn as nn


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=16):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0

        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # GroupNorm针对 (channels//self.groups) 通道
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)

        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups,
                                 kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups,
                                 kernel_size=3, stride=1, padding=1)

        # ----- 额外新增的，用于存储本次 forward 的注意力 -----
        self.last_weights = None

    def forward(self, x):
        b, c, h, w = x.size()

        # 1) 将 x 分组: (B*g, c//g, H, W)
        group_x = x.reshape(b * self.groups, -1, h, w)

        # 2) 分别在 H/W 方向做池化
        x_h = self.pool_h(group_x)  # (B*g, c//g, H, 1)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # (B*g, c//g, 1, W)

        # 3) 用1x1卷积融合 H/W 信息
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 分成 (H, 1) 和 (W, 1)

        # 4) x1 = group_x * x_h * x_w  (再做GroupNorm)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        # 5) x2 = 3x3卷积  (进一步特征提取)
        x2 = self.conv3x3(group_x)

        # 6) 全局平均池化 + softmax 得到注意力
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        # 7) 计算加权 (B*g, 1, H, W)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        weights_sig = weights.sigmoid()

        # -------- 新增：将本次的注意力图存到 self.last_weights，便于外部读取 --------
        self.last_weights = weights_sig.detach()

        # 8) 用注意力加权 group_x，并 reshape 回 (B, C, H, W)
        out = (group_x * weights_sig).reshape(b, c, h, w)
        return out
