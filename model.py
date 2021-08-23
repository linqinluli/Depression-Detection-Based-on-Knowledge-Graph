import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    '''
    Add an attention layer to judge the importance of each property
    P: input property
    P': Output property
    P' = P x aT
    a = softmax(PT x W1 + b1)
    '''
    def __init__(self, in_features):
        super(AttentionLayer, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(in_features, in_features))

        self.bias = nn.Parameter(torch.Tensor(1, in_features))

    def forward(self, input):
        
        output =torch.zeros_like(input)
        for i in range(input.shape[0]):

            alpha = F.softmax(torch.mm(input[i].t(), self.weight) + self.bias)
            res = input[i] * alpha.t()
            output[i] = res

        return input


class DEPredictor(torch.nn.Module):
    def __init__(self, property_num):
        super(DEPredictor, self).__init__()

        self.attention = AttentionLayer(property_num)
        self.fn1 = nn.Linear(property_num, property_num)
        self.fn2 = nn.Linear(property_num, 1)

    def forward(self, x):

        # print(x.shape)
        x = self.attention(x)
        x = torch.reshape(x, (x.shape[0], -1))
        # print(x.shape)
        x = self.fn1(x)
        x = self.fn2(x)

        output = F.sigmoid(x)

        return output