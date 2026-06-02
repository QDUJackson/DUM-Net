import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import dct, idct
import numpy as np
from module.activations import *
import time
from module.activations import *
# 瀹炵幇2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


# 瀹炵幇2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')


# 鐢熸垚2D DCT鍗风Н鏍?
def gen_dct2(n):
    C = np.zeros([n ** 2, n ** 2])
    for i in range(n):
        for j in range(n):
            A = np.zeros([n, n])
            A[i, j] = 1
            B = idct2(A)
            # 灏嗙敓鎴愮殑鍗风Н鏍稿瓨鍒扮煩闃礐涓?
            # 鏍煎紡鍖栬緭鍑築鐨勬瘡涓€琛?
            '''formatted_B = [["{:.2e}".format(b) for b in row] for row in B]
            print(np.array(formatted_B))  # 鎵撳嵃鏍煎紡鍖栧悗鐨凚'''
            C[:, j * n + i] = B.reshape(-1)



    return C


class TNRDConv2d(nn.Module):
    """TNRD鍗风Н灞傦紝浣跨敤DCT鍗风Н鏍稿苟寮曞叆鍙涔犵殑鍔犳潈缁撴瀯"""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, shold_values=0.12,bias=True):

        super(TNRDConv2d, self).__init__()



        self.act = RBF(63, out_channels, triangular)

        '''self.act_value = shold_values
        self.act = ReLUShold(0.12)'''
        
        # 鐢熸垚DCT鍗风Н鏍?
        dct_matrix = torch.Tensor(gen_dct2(kernel_size))  # 鐢熸垚鐨勬槸 (n^2, n^2) 澶у皬鐨勭煩闃?
        dct_kernels = []  # 鐢ㄦ潵瀛樺偍姣忎釜鎭㈠鍑烘潵鐨勫嵎绉牳

        dct_matrix = dct_matrix[:, 1:]  # 鍘绘帀鐩存祦鍒嗛噺锛屽搴斿幓闄ょ煩闃电殑绗竴鍒?

        for i in range(out_channels):
            channel_kernels = []  # 閽堝姣忎釜杈撳嚭閫氶亾
            for j in range(in_channels):
                kernel_idx = i * in_channels + j  # 璁＄畻褰撳墠绱㈠紩
                if kernel_idx < dct_matrix.size(1):  # 纭繚绱㈠紩涓嶈秴鐣?
                    kernel = dct_matrix[:, kernel_idx].view(kernel_size, kernel_size)
                    channel_kernels.append(kernel)

            dct_kernels.append(channel_kernels)


        # 灏嗘瘡涓嵎绉牳鍫嗗彔鎴愪竴涓?tensor锛屽舰鐘朵负 (out_channels, in_channels, kernel_size, kernel_size)
        dct_kernel_total = torch.stack([torch.stack(k) for k in dct_kernels])

        # 杞崲涓哄紶閲?
        self.dct_kernels = nn.Parameter(dct_kernel_total, requires_grad=False)

        # 鍙涔犵殑鍔犳潈鍙傛暟锛屽垵濮嬪寲涓哄叏 1锛岀劧鍚庨€氳繃 L2 褰掍竴鍖?
        self.weights = nn.Parameter(torch.ones(out_channels, out_channels), requires_grad=True)
        nn.init.normal_(self.weights, mean=0.0, std=0.01)

        '''
        杩欓噷鍙渶瑕?m^2 - 1鍙涔犵殑鍙傛暟
        '''

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, input):

        u = input

        # 娣诲姞瀵圭О锛堝弽灏勶級褰㈠紡鐨刾adding
        padded_input = F.pad(u, pad=[self.padding, self.padding, self.padding, self.padding], mode='reflect')

        # 瀵瑰彲瀛︿範鐨勬潈閲嶈繘琛?L2 褰掍竴鍖?
        l2_norm = torch.norm(self.weights, p=2, dim=1, keepdim=True)
        normalized_weights = self.weights / (l2_norm + 1e-10)

        # 瀵规瘡涓嵎绉牳杩涜鍔犳潈
        dct_kernels_swapped = self.dct_kernels.transpose(0, 1)

        weighted_kernels = dct_kernels_swapped * normalized_weights.unsqueeze(2).unsqueeze(3)
        weighted_kernels = weighted_kernels.sum(dim=1, keepdim=True)

        # 杩涜鍗风Н鎿嶄綔
        output = F.conv2d(padded_input, weighted_kernels, stride=self.stride, padding=self.padding, dilation=self.dilation)
        output_act = self.act(output)

        # 瀵瑰嵎绉牳杩涜180搴﹀弽杞?
        weight_rot180 = torch.rot90(torch.rot90(weighted_kernels, 1, [2, 3]), 1, [2, 3])
        # 浣跨敤 groups 鍙傛暟锛岀‘淇濇瘡涓€氶亾鐙珛杩涜鍗风Н

        output180 = F.conv2d(output_act, weight_rot180, stride=self.stride, padding=self.padding, dilation=self.dilation,groups=output_act.shape[1])

        '''
        groups鍐欐垚output_act.shape[1]涔嬪悗,鏁翠綋鐨勫弽鍗风Н灏卞彉鎴愪簡娣卞害鍗风Н
        鎵€浠ヨ繖涓湴鏂瑰啓鐨勫嵎绉牳鐨勫舰鐘跺氨鏄痆24,1,5,5]浜嗭紝铏界劧鍙嶅嵎绉殑杈撳叆閫氶亾鏄?4
        '''
        # 鍘婚櫎padding锛屾仮澶嶅埌鍘熸潵鐨勫昂瀵?
        output180 = output180[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return output180


# 瀹炰緥鍖朤NRDConv2d
if __name__ == "__main__":
    tnrd_conv2d = TNRDConv2d(1, 8, kernel_size=3, stride=1, padding=1, dilation=1)
    input_tensor = torch.randn(8, 1, 64, 64)
    output = tnrd_conv2d(input_tensor)
    print("output shape:", output.shape)
