import torch
import torch.nn as nn
from module.ExtractFeatures import ExtractFeatures  # 或者把ExtractFeatures放在同一文件
from module.pdfNet import distNetwork


class PseudoSiameseNet(nn.Module):
    """
    使用两份 ExtractFeatures，分别处理两种模态的输入，实现“伪孪生网络”。
    每份模型都拥有独立的参数。
    """

    def __init__(self):
        super(PseudoSiameseNet, self).__init__()
        # 分别实例化两份独立的ExtractFeatures
        self.extractor_t1 = ExtractFeatures()  # 用于处理模态1 (T1)
        self.extractor_t2 = ExtractFeatures()  # 用于处理模态2 (T2)

        self.back_f1 = ExtractFeatures()
        self.back_f2 = ExtractFeatures()

    def forward(self, x_t1, x_t2):
        """
        x_t1: [N, 1, H, W]  T1模态灰度图
        x_t2: [N, 1, H, W]  T2模态灰度图
        return: (feat_t1, feat_t2) 分别是网络输出结果
        """
        feat_t1 = self.extractor_t1(x_t1)
        feat_t2 = self.extractor_t2(x_t2)

        feat_f1 = self.back_f1(feat_t1)
        feat_f2 = self.back_f2(feat_t2)

        return feat_t1, feat_t2,feat_f1,feat_f2
