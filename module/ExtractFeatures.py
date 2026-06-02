import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    卷积块：卷积 -> 实例归一化（可选） -> Leaky ReLU
    """

    def __init__(self, in_channels, out_channels, negative_slope=0.2, use_instancenorm=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.use_instancenorm = use_instancenorm
        if self.use_instancenorm:
            self.inorm = nn.InstanceNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_instancenorm:
            x = self.inorm(x)
        x = self.lrelu(x)
        return x


class ExtractFeatures(nn.Module):
    """
    ExtractFeatures 网络模型，类似于 U-Net 架构
    """

    def __init__(self):
        super(ExtractFeatures, self).__init__()

        # 下采样路径
        #1
        self.conv1 = ConvBlock(1, 32, negative_slope=0.2, use_instancenorm=True)  # 输入通道1，输出通道32
        #2
        self.conv2 = ConvBlock(32, 32, negative_slope=0.2, use_instancenorm=True)  # 输入通道32，输出通道32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样
        #3
        self.conv3 = ConvBlock(32, 64, negative_slope=0.2, use_instancenorm=True)  # 输入通道32，输出通道64
        #4
        self.conv4 = ConvBlock(64, 64, negative_slope=0.2, use_instancenorm=True)  # 输入通道64，输出通道64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样

        #5
        # 上采样路径
        # 使用nearest保持图像边缘清晰
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)  # 上采样
        self.conv5 = ConvBlock(64, 64, negative_slope=0.2, use_instancenorm=False)  # 上采样后卷积（无需实例归一化）

        #6
        # conv6 接收拼接后的特征图，通道数为 64 (up1) + 64 (skip2) = 128
        self.conv6 = ConvBlock(96, 64, negative_slope=0.2, use_instancenorm=True)  # 拼接后卷积（64+64=128 通道）

        #7
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)  # 上采样
        self.conv7 = ConvBlock(64, 64, negative_slope=0.2, use_instancenorm=False)  # 上采样后卷积（无需实例归一化）

        #8
        # conv8 接收拼接后的特征图，通道数为 64 (up2) + 32 (skip1) = 96
        self.conv8 = ConvBlock(96, 64, negative_slope=0.2, use_instancenorm=True)  # 拼接后卷积（64+32=96 通道）

        #9
        self.conv9 = ConvBlock(64, 32, negative_slope=0.2, use_instancenorm=True)
        # 输出层
        #10
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3,stride=1,padding=1)  # 输出通道1

    def forward(self, x):
        # 下采样路径
        #1
        x1 = self.conv1(x)  # (N, 32, H, W)
        skip1 = x1  # 保存跳接 (N, 32, H, W)

        #2
        x2 = self.conv2(x1)  # (N, 32, H, W)
        p1 = self.pool1(x2)  # (N,32,H/2,W/2)
        skip2 = p1

        #3
        x3 = self.conv3(p1)  # (N,64,H/2,W/2)

        #4
        x4 = self.conv4(x3)  # (N,64,H/2,W/2)
        p2 = self.pool2(x4)  # (N,64,H/4,W/4)

        #5
        # 上采样路径
        up1 = self.up1(p2)  # (N,64,H/2,W/2)
        up1 = self.conv5(up1)  # (N,64,H/2,W/2) 无实例归一化

        #6
        # 拼接跳接2 (x4) 和 up1
        concat1 = torch.cat([up1, skip2], dim=1)  # (N,64+64=128,H/2,W/2)
        x5 = self.conv6(concat1)  # (N,64,H/2,W/2) 使用实例归一化

        #7
        up2 = self.up2(x5)  # (N,64,H,W)
        up2 = self.conv7(up2)  # (N,64,H,W) 无实例归一化

        #8
        # 拼接跳接1 (skip1) 和 up2
        concat2 = torch.cat([up2, skip1], dim=1)  # (N,64+32=96,H,W)
        x6 = self.conv8(concat2)  # (N,32,H,W) 使用实例归一化

        #9
        x7 = self.conv9(x6)


        #10
        # 输出层
        output = self.final_conv(x7)  # (N,1,H,W)


        return output


# 测试程序
'''if __name__ == "__main__":
    # 创建模型实例
    model = ExtractFeatures()

    # 创建一个随机输入张量，形状为 (批量大小, 通道数, 高度, 宽度)
    batch_size = 8
    channels = 1
    height = 256
    width = 256
    input_tensor = torch.randn(batch_size, channels, height, width)
    print(f"\n输入张量形状: {input_tensor.shape}")

    # 将模型设置为评估模式
    model.eval()

    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
    print(f"输出张量形状: {output.shape}")  # 输出形状应为 (1, 1, 128, 128)
'''