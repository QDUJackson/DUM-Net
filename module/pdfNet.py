import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy

class distNetwork(nn.Module):

    def __init__(self,beta,device,NG = 256):
        super(distNetwork, self).__init__()
        #self.size = size
        # this size is a batch contained size
        self.NG = NG
        # the NG is bins num
        '''
        数值范围在0-1内时      beta = gaussian sigma = 0.0314
        数值范围在0-255内时    beta = gaussian sigma = 8
        '''

        self.beta = beta
        # the beta is for gaussian kernel function

        self.device = device

        self.bins_grid = torch.linspace(0,self.NG - 1,self.NG).to(self.device)
        # To create a tensor from 0 - NG-1 and NG points totally
        ''' 
        下面的 bins_grid / 255.0 是为了让模型在0-1的数据范围内
        '''
        self.bins_grid = self.bins_grid / 255.0
        self.bins_grid = self.bins_grid.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # unsqueeze(0) means add a new dimension in dim=0
        # so the shape of bins_grid is [1,1,1,256]

        '''self.batch_size = size[0]
        self.channels = size[1]
        self.height = size[2]
        self.width = size[3]'''

    def forward(self,I1,I2):

        I1_flat = I1.reshape(I1.shape[0],I1.shape[1],I1.shape[2] * I1.shape[3],1)
        I2_flat = I2.reshape(I1.shape[0],I1.shape[1],I1.shape[2] * I1.shape[3],1)
        # shape is [batch_size,channels,height * width, 1]
        I1_value = I1_flat - self.bins_grid
        I2_value = I2_flat - self.bins_grid
        # The shape of I1_flat is [batch_size,channels,height * width,1]
        # The shape of bins_grid is [1,1,1,256]
        # So it is a broadcasting
        # I1_value shape is [batch_size, channels, height*width, 256]
        I1_value_gauss = self.gaussian_function_with_coefficient(I1_value,self.beta).permute(0,1,3,2)
        # permute for matmul
        I2_value_gauss = self.gaussian_function_with_coefficient(I2_value,self.beta)

        # So the shape of I2_value_gauss is [batch_size,channels,height * width,256]
        # I1_value_gauss's shape is [batch_size,channels,256,height * width]

        H_temp = torch.matmul(I1_value_gauss,I2_value_gauss)
        H = H_temp / H_temp.sum(dim=(1,2,3),keepdim=True)
        return H

    def gaussian_function_with_coefficient(self,x, beta):
        # 确保 beta 是一个张量
        beta_tensor = torch.tensor(beta, dtype=torch.float32, device=x.device)  # 确保 beta 在同一个设备上
        one_tensor = torch.ones_like(beta_tensor)
        coefficient = 1 / ( torch.sqrt(2 * torch.pi * one_tensor) * beta_tensor)
        exponent = torch.exp(- (x ** 2) / (2 * beta_tensor ** 2))
        result = coefficient * exponent
        return result