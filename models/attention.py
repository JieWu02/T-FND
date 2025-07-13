import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Dot_Attention(nn.Module):

    def __init__(self, query_dim, key_dim, num_units): #（64，64，64）
        super().__init__()
        self.num_units = num_units
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key, mask=None):
        querys = self.W_query(query)
        keys = self.W_key(key)
        values = self.W_value(key)

        scores = torch.matmul(querys, keys.transpose(-2, -1))
        scores = scores / (self.key_dim ** 0.5)

        scores = F.softmax(scores, dim=-1)
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        return out