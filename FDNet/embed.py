import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv2d(in_channels=1, out_channels=d_model,
                                   kernel_size=1, padding=0)

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 3, 1, 2))  # B, C, L ,V
        return x


class DataEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        return self.dropout(x)
