from module.UNetNormalClass import U_Network
from module.Cycle_extractFeatures import PseudoSiameseNet
import torch.nn as nn
from module.utils import warp_images_grid_sample  # 假设 warp_images_grid_sample 在这里
import torch


class RegistrationModel(nn.Module):
    def __init__(self,
                 dim=2,
                 enc_nf=[16, 32, 32, 32],
                 dec_nf=[32, 32, 32, 32, 32, 16],
                 bn=False,
                 full_size=True,
                 siamese_weight_path="model_pth/best_model_Extract.pth",
                 device= 'cuda:0'):
        """
        :param dim: 维度 (2D/3D)，此处默认为 2D
        :param enc_nf: U-Net 编码器的通道数
        :param dec_nf: U-Net 解码器的通道数
        :param bn: 是否使用 batchnorm
        :param full_size: 是否在 full_size 再多做一次卷积
        :param siamese_weight_path: 伪孪生网络的预训练权重路径
        """
        super(RegistrationModel, self).__init__()
        # 1) 伪孪生网络 (不参与训练)
        self.siamese = PseudoSiameseNet()
        # 加载预训练好的伪孪生网络权重
        state_dict = torch.load(siamese_weight_path, map_location=device)
        self.siamese.load_state_dict(state_dict)
        # 冻结伪孪生网络参数，使其不参与训练
        for param in self.siamese.parameters():
            param.requires_grad = False

        # 2) U-Net 用来生成变形场 (参与训练)
        self.unet = U_Network(dim=dim, enc_nf=enc_nf, dec_nf=dec_nf, bn=bn, full_size=full_size)

    def forward(self, t1_img, t2_img):
        """
        :param t1_img: [N, 1, H, W] - T1 模态图像
        :param t2_img: [N, 1, H, W] - T2 模态图像
        :return: (warped_t1_img, feat_t2)
        """
        # 1) 使用预训练好的 PseudoSiameseNet 得到共享空间表示 (不会更新参数)
        feat_t1, feat_t2, feat_f1, feat_f2 = self.siamese(t1_img, t2_img)

        # 2) 将共享空间下的 T1 & T2 输入 U-Net，得到变形场 flow
        #    flow.shape = [N, 2, H_feat, W_feat]
        #    假设此处 feat_t1, feat_t2 的 spatial size 和原图一致
        #flow = self.unet(feat_t1, feat_t2)
        flow = self.unet(t1_img,t2_img)
        '''
        这个地方影响很大
        '''
        # 假设我们把 flow 拆成 x/y 两个分量，以便在 warp_images_grid_sample 中使用
        flow_x, flow_y = torch.split(flow, 1, dim=1)

        # 3) 应用变形场到共享空间下的 T1 特征 (同理可应用到原图或其他图像)
        feat_t1_warpped = warp_images_grid_sample(feat_t1, flow_x * 255.0, flow_y * 255.0)

        return feat_t1_warpped, feat_t1, feat_t2, flow


'''if __name__ == "__main__":
    # 简单测试一下
    t1_img1 = torch.randn(2, 1, 128, 128)
    t2_img = torch.randn(2, 1, 128, 128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # 实例化注册模型 (只需要修改 siamese_weight_path 为您实际的路径即可)
    model = RegistrationModel(
        dim=2,
        enc_nf=[16, 32, 32, 32],
        dec_nf=[32, 32, 32, 32, 32, 16,16],
        bn=False,
        full_size=True,
        siamese_weight_path="../model_pth/best_model.pth",
        device = device
    )

    # 前向推理
    warped_t1, feat_t2 = model(t1_img1, t2_img)
    print("warped_t1.shape:", warped_t1.shape)  # 预期形状 [N, C, H, W]
    print("feat_t2.shape:", feat_t2.shape)      # 预期形状 [N, C, H, W]
'''