# -*- coding: UTF-8 -*-
import torch


class Config():
    def __init__(self):
        self.maxlength = 30  # 句子最大长度
        self.batch_size = 64
        self.lr = 0.0001
        self.d_model = 512  # Attention的维度
        self.d_ff = 2048  # 全连接层的维度
        self.n_layers = 6
        self.heads = 8
        self.dropout = 0.2
        self.n_epochs = 60
        self.PAD_token = 0
        self.UNK_token = 1
        self.SOS_token = 2
        self.EOS_token = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


