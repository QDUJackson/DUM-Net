from module.pdfNet import distNetwork
from module.Normal_DCTConv2D import TNRDConv2d
import torch.nn.functional as F
import torch
import torch.nn as nn
from module.utils import *
from module.model_single_swin_No_EMA import ViTVNet

import module.configs as configs

from module.Registration_Total_View import RegistrationModel
import time
from losses import dice_coefficient

from module.networks import VxmDense  # 请确保 VxmDense 类的实现文件路径正确
from module.myconfigs import unet_features_2d


class ElasticityRegularizer(nn.Module):
    def __init__(self, kesai, device):
        super(ElasticityRegularizer, self).__init__()
        self.kesai = nn.Parameter(torch.Tensor([kesai]).to(device))

    def forward(self, fai_init1, fai_init2):
        # fai_init1是竖直方向， fai_init2是水平方向
        kesai = torch.sigmoid(self.kesai)
        fai_init1_padding = F.pad(fai_init1, (1, 1, 1, 1), mode='replicate')
        fai_init2_padding = F.pad(fai_init2, (1, 1, 1, 1), mode='replicate')

        laplace_1 = (
                fai_init1_padding[:, :, 2:, 1:-1] +  # 下
                fai_init1_padding[:, :, :-2, 1:-1] +  # 上
                fai_init1_padding[:, :, 1:-1, 2:] +  # 右
                fai_init1_padding[:, :, 1:-1, :-2] -  # 左
                4.0 * fai_init1_padding[:, :, 1:-1, 1:-1]  # 中间
        )
        laplace_2 = (
                fai_init2_padding[:, :, 2:, 1:-1] +  # 下
                fai_init2_padding[:, :, :-2, 1:-1] +  # 上
                fai_init2_padding[:, :, 1:-1, 2:] +  # 右
                fai_init2_padding[:, :, 1:-1, :-2] -  # 左
                4.0 * fai_init2_padding[:, :, 1:-1, 1:-1]  # 中间
        )

        grad_2 = (
                fai_init2_padding[:, :, 1:-1, :-2] +  # 左
                fai_init2_padding[:, :, 1:-1, 2:] -  # 右
                2.0 * fai_init2_padding[:, :, 1:-1, 1:-1]  # 中间
        )

        grad_1 = (
                fai_init1_padding[:, :, :-2, 1:-1] +  # 上
                fai_init1_padding[:, :, 2:, 1:-1] -  # 下
                2.0 * fai_init1_padding[:, :, 1:-1, 1:-1]  # 中间
        )

        round_1 = (
                fai_init1_padding[:, :, :-2, :-2] -  # 左上 (i-1,j-1)
                fai_init1_padding[:, :, 2:, :-2] -  # 左下 (i+1,j-1)
                fai_init1_padding[:, :, :-2, 2:] +  # 右上 (i-1,j+1)
                fai_init1_padding[:, :, 2:, 2:]  # 右下 (i+1,j+1)
        )
        round_2 = (
                fai_init2_padding[:, :, :-2, :-2] -  # 左上 (i-1,j-1)
                fai_init2_padding[:, :, 2:, :-2] -  # 左下 (i+1,j-1)
                fai_init2_padding[:, :, :-2, 2:] +  # 右上 (i-1,j+1)
                fai_init2_padding[:, :, 2:, 2:]  # 右下 (i+1,j+1)
        )

        '''Result1 = self.kesai * laplace_1 + (1.0 - self.kesai) * grad_1 + 0.25 * (1.0 - self.kesai) * round_2
        Result2 = self.kesai * laplace_2 + (1.0 - self.kesai) * grad_2 + 0.25 * (1.0 - self.kesai) * round_1'''
        Result1 = kesai * laplace_1 + (1.0 - kesai) * grad_1 + 0.25 * (1.0 - kesai) * round_2
        Result2 = kesai * laplace_2 + (1.0 - kesai) * grad_2 + 0.25 * (1.0 - kesai) * round_1
        return Result1, Result2


class TNRDLayer(nn.Module):
    def __init__(self, shold_values, device):
        super(TNRDLayer, self).__init__()
        # 定义 TNRDConv2d 的两个分支

        self.elasticity_regular = ElasticityRegularizer(-2.2, device)

        # 定义 alpha_grad 作为 nn.Parameter 并指定设备
        '''self.alpha_grad = nn.Parameter(torch.Tensor([0.9]).to(device))
        self.dt = nn.Parameter(torch.Tensor([1.01]).to(device))'''
        # self.alpha_grad = nn.Parameter(torch.Tensor([torch.log(torch.Tensor([0.9]))]).to(device))
        self.dt = nn.Parameter(torch.Tensor([torch.log(torch.Tensor([1e-3]))]).to(device))
        self.alpha_grad = 1.0
        self.elas = 0.05

    def forward(self, fai_init1, fai_init2, fai_unet_1, fai_unet_2):
        # 通过 TNRDConv 分支
        # fai_init1和fai_init2是两分支前面的输入
        # dt = torch.relu(self.dt)

        dt = torch.exp(self.dt)  # 确保 dt 始终为正
        # alpha_grad = torch.exp(self.alpha_grad)  # 确保 alpha_grad 始终为正
        alpha_grad = self.alpha_grad

        fai_elasticity_1, fai_elasticity_2 = self.elasticity_regular(fai_init1, fai_init2)
        # print(f"fai elasticity mul elas max is {(fai_elasticity_1 * self.elas).max()}")
        # 其中fai_unet_1和fai_unet_2都是unet分支的产物
        # 汇聚两个分支进行梯度下降更新
        # print(f"fai_unet mul alpha_grad max is {(self.alpha_grad * fai_unet_1).max()}")

        '''fai_result_1 = fai_init1 - dt * (self.elas * fai_elasticity_1 + self.alpha_grad * fai_unet_1)
        fai_result_2 = fai_init2 - dt * (self.elas * fai_elasticity_2 + self.alpha_grad * fai_unet_2)'''

        fai_result_1 = fai_init1 - dt * (self.elas * fai_elasticity_1 + alpha_grad * fai_unet_1)
        fai_result_2 = fai_init2 - dt * (self.elas * fai_elasticity_2 + alpha_grad * fai_unet_2)

        '''print(f"dt is {dt}")
        print(f"fai_unet1 is {fai_unet_1.mean()}")
        print(f"fai_elasticity_1 is {fai_elasticity_1.mean()}")'''

        return fai_result_1, fai_result_2


class UTNet(nn.Module):
    def __init__(self, beta, enc_nf, dec_nf, size, device, size_tensor, num_layers=5, shold_values=0.16):
        super(UTNet, self).__init__()
        self.num_layers = num_layers
        CONFIGS = {
            'ViT-V-Net': configs.get_2DReg_config(),
        }
        # 共享的 U-Net 分支
        config = CONFIGS['ViT-V-Net']
        self.unet = ViTVNet(config, img_size=(256, 256))

        self.Get_Histogram = distNetwork(beta, device)

        self.batch_size = size_tensor[0]
        '''
            修改一下初始化
        '''
        fai_init = 0.1 + 0.1 * torch.rand(self.batch_size, 2, 256, 256).to(device=device)
        fai_init = fai_init / 255.0

        fai_init1, fai_init2 = torch.split(fai_init, 1, dim=1)

        self.register_buffer('fai_init', fai_init)
        self.register_buffer('fai_init1', fai_init1)
        self.register_buffer('fai_init2', fai_init2)

        # 使用 nn.ModuleList 来存放多个独立的 TNRDLayer，并为每个层定义独立的 alpha_grad
        self.tnrd_layers = nn.ModuleList([TNRDLayer(shold_values, device) for _ in range(num_layers)])
        self.elas = self.tnrd_layers[0].elas

        # reg unet
        self.pre_reg = VxmDense(
            inshape=(256, 256),  # 2D 输入大小
            nb_unet_features=unet_features_2d,  # 使用默认多层 Unet 配置
            nb_unet_levels=None,
            unet_feat_mult=1,
            nb_unet_conv_per_level=1,
            int_steps=0,  # 非 diffeomorphic（不做积分）
            int_downsize=2,
            bidir=False,
            src_feats=1,
            trg_feats=1,
            unet_half_res=False
        ).to(device)
        # state_dict = torch.load("model_unet_pth/best_model_2025036_1.pth",map_location=device)
        self.model_path = "Vxm_Path/best_dice_model_vxm.pth"
        state_dict = torch.load(self.model_path, map_location=device)
        '''
        使用的模型影响也很大
        '''
        self.pre_reg.load_state_dict(state_dict)
        for param in self.pre_reg.parameters():
            param.requires_grad = False

    def forward(self, I1, I2, seg):
        # 初始化
        # 这个I2_init就是模型图片形式的输入

        # I2 代表模态1 I1 代表模态2

        feat_t1_warrped, flow = self.pre_reg(I1, I2)

        flow = flow / 255.0

        self.fai_init1, self.fai_init2 = torch.split(flow, 1, dim=1)

        self.fai_init = flow

        I1_init = warp_images_grid_sample(I1, self.fai_init1 * 255.0, self.fai_init2 * 255.0)

        fai_result1, fai_result2 = self.fai_init1, self.fai_init2
        fai_result = torch.cat((fai_result1, fai_result2), dim=1)

        dice_list = []

        # 通过每个层（共享 U-Net 和独立 TNRDLayer）进行计算
        for tnrd_layer in self.tnrd_layers:
            # 使用共享的 U-Net 进行计算
            I_cat = torch.cat((I1_init, I2), dim=1)
            # I2_init是source, I1是target
            fai = self.unet(I_cat)
            fai_unet_1, fai_unet_2 = torch.split(fai, 1, dim=1)

            # 使用当前层的 TNRDLayer 进行计算
            fai_result_1, fai_result_2 = tnrd_layer(fai_result1, fai_result2, fai_unet_1, fai_unet_2)

            '''
            fai_result是每一层变形场合起来的形式
            fai_result1,fai_result2是每一层变形场分离的形式
            I2_init是每一层图片形式的输入
            '''

            # 将结果拼接起来
            fai_result = torch.cat((fai_result_1, fai_result_2), 1)

            # 更新 I2_init 以用于下一个层的计算
            I1_init = warp_images_grid_sample(I1, fai_result_1 * 255.0, fai_result_2 * 255.0)
            # 注意 F.grid_sample的摆放顺序是（水平方向，竖直方向）
            # 但warp函数的输入顺序是相反的  （竖直方向，水平方向）
            # 更新 fai_result1 和 fai_result2
            fai_result1, fai_result2 = torch.split(fai_result, 1, dim=1)

        I1w = I1_init
        # print(f'seg in train is {(seg == 64.0 / 255.0).sum().item()}')

        seg_source_idx = label_value_to_index(seg * 255.0)
        seg_source_1h = label_to_one_hot(seg_source_idx, num_classes=4)
        seg_wrapped = warp_images_grid_sample(seg_source_1h, fai_result1 * 255.0, fai_result2 * 255.0)
        # 假设 seg_wrapped 的形状为 [B, 4, H, W]
        seg_idx = torch.argmax(seg_wrapped, dim=1)  # 形状变为 [B, H, W]
        # 定义映射数组
        mapping = torch.tensor([0, 64, 128, 255], device=seg_idx.device)
        # 利用 mapping 将类别索引映射为对应的离散值
        seg_discrete = mapping[seg_idx].unsqueeze(1)

        # I1_patches,I2_patches = extract_dynamic_patches_with_grid_sample(I1w,I2,seg_wrapped)

        # 计算全局直方图
        # H = self.Get_Histogram(I1w, I2)
        # H_patches = self.Get_Histogram(I1_patches, I2_patches)

        return I1w, fai_result, seg_discrete


'''def test_utnet():
    # Define constants
    beta = 0.1
    enc_nf, dec_nf = 16, 16
    img_size = (256, 256)
    batch_size = 2
    num_layers = 3

    # Device configuration
    device = torch.device("cpu")

    # Create synthetic input tensors
    I1 = torch.rand(batch_size, 1, *img_size).to(device)  # Target image
    I2 = torch.rand(batch_size, 1, *img_size).to(device)  # Source image
    seg = torch.randint(0, 4, (batch_size, 1, *img_size)).float().to(device) / 3.0  # Segmentation map (normalized)

    # Define the UTNet model
    size_tensor = (batch_size, 1, *img_size)
    utnet = UTNet(beta=beta, enc_nf=enc_nf, dec_nf=dec_nf, size=img_size, device=device, size_tensor=size_tensor, num_layers=num_layers)
    utnet = utnet.to(device)

    # Forward pass
    I2w, fai_result, H, H_patches, seg_wrapped_discrete = utnet(I2, I1, seg)

    # Assertions to validate outputs
    assert I2w.shape == I1.shape, f"Warped image shape mismatch: {I2w.shape} vs {I1.shape}"
    assert fai_result.shape == (batch_size, 2, *img_size), f"Deformation field shape mismatch: {fai_result.shape}"
    assert seg_wrapped_discrete.shape == seg.shape, f"Wrapped segmentation shape mismatch: {seg_wrapped_discrete.shape}"

# Run the test function
test_utnet()'''