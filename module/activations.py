import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
import time
class ReLUShold(nn.Module):
    def __init__(self, init_t=0.1):
        super(ReLUShold, self).__init__()
        # 灏?t 瀹氫箟涓哄彲瀛︿範鍙傛暟
        self.t = nn.Parameter(torch.tensor(init_t, dtype=torch.float32))

    def forward(self, x):

        return  F.relu((x - self.t) ,inplace=True) -   F.relu(-(x + self.t), inplace=True)

class CustomActivation(nn.Module):
    def __init__(self, init_t=2.5, ep=0.5):

        super(CustomActivation, self).__init__()
        # 灏?t 瀹氫箟涓哄彲瀛︿範鍙傛暟
        self.t = nn.Parameter(torch.tensor(init_t, dtype=torch.float32))
        self.ep = ep

    def Tanhsoft1(self,x):

        return torch.log(1+torch.exp(x)) * torch.tanh(x)

    def forward(self, x):
        # 璁＄畻鍏紡锛?        # ep * log(1+exp((x-t)/ep)) * tanh((x-t)/0.5)
        # - ep * log(1+exp(-(x+t)/ep)) * tanh(-(x+t)/0.5)
        return self.ep * self.Tanhsoft1((x-self.t)/self.ep) - self.ep * self.Tanhsoft1(-(x + self.t) / self.ep)



# RBF Layer
class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        num_func: Number of radial basis functions (M in the formula)
        num_filters: Number of input filters (channels of the input)
        basis_func: Radial basis function to apply (Gaussian, triangular, etc.)
    """

    def __init__(self, num_func, num_filters, basis_func):
        super(RBF, self).__init__()

        self.num_func = num_func
        self.num_filters = num_filters
        self.basis_func = basis_func

        # 娉ㄥ唽涓績鐐癸紝鍒濆鑼冨洿[-0.5, 1.5]
        self.register_buffer('centers', torch.linspace(-310, 310, num_func))

        # 鍙涔犵殑鏉冮噸鍙傛暟 (num_func, num_filters) -> 姣忎釜閫氶亾閮芥湁鐙珛鐨勬潈閲?        self.weight = nn.Parameter(torch.Tensor(num_func, num_filters))

        # 鍒濆鍖栧弬鏁?        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # 璁剧疆 gamma 涓哄父鏁?        self.gamma = 10


    def forward(self, x):
        # 鎵╁睍杈撳叆鐨勭淮搴︿互杩涜骞挎挱锛?batch_size, channels, height, width) -> (batch_size, channels, num_func, height, width)
        x = x.unsqueeze(2) * 255.0
        print(x.shape)
        centers = self.centers.view(1, 1, self.num_func, 1, 1)
        # 灏?centers 鎵╁睍缁村害浠ヤ究涓庤緭鍏ュ箍鎾紝(num_func) -> (1, 1, num_func, 1, 1)
        print(centers.shape)
        diff = torch.abs(x - centers)
        print(diff.shape)
        rbf_out = self.basis_func(diff, self.gamma)
        # 灏?RBF 杈撳嚭涓庢潈閲嶇浉涔橈紝鏉冮噸 shape 涓?(num_func, num_filters)
        # 鎵╁睍鏉冮噸鐨勭淮搴︿互鍖归厤杈撳叆锛?num_func, num_filters) -> (1, num_filters, num_func, 1, 1)
        print(rbf_out.shape)
        weighted_rbf_out = rbf_out * self.weight.view(1, self.num_filters, self.num_func, 1, 1)
        print(weighted_rbf_out.shape)
        # 鍦?num_func 缁村害涓婅繘琛屾眰鍜岋紝(batch_size, channels, height, width)
        output = weighted_rbf_out.sum(dim=2)
        print(output.shape)
        output_norm = torch.clamp(output, 0, 255.0)

        return output_norm / 255.0

class SoftThresholdActivation(nn.Module):
    def __init__(self, init_lambda=2):
        super(SoftThresholdActivation, self).__init__()
        # 瀹氫箟闃堝€间负鍙涔犵殑鍙傛暟锛屽垵濮嬪€间负 init_lambda
        self.lambda_ = nn.Parameter(torch.tensor(init_lambda, dtype=torch.float32))

    def forward(self, x):
        # 瀹炵幇杞槇鍊兼搷浣滐紝纭繚 lambda_ 涓烘鍊?        ep = 0.5
        t = torch.relu(self.lambda_)
        return ep*torch.log(1+torch.exp((x-t)/ep))-ep*torch.log(1+torch.exp(-(x+t)/ep))


# RBFs

def triangular(alpha, gamma):
    out = 1 - alpha.div(gamma)
    out[alpha > gamma] = 0
    return out

def gaussian(alpha, gamma):
    out = torch.exp(-alpha.pow(2) / (2 * gamma.pow(2)))
    return out

# 瀹炰緥鍖朢BF灞?tnrd_RBF = RBF(63, 8, triangular)

# 娴嬭瘯
if __name__ == "__main__":
    tnrd_RBF = RBF(63, 8, triangular)
    input_tensor = torch.randn(8, 8, 64, 64)
    output = tnrd_RBF(input_tensor)
    print("output shape:", output.shape)
