import torch
import torch.nn as nn
import math
from torch.nn.utils import weight_norm


class ConvLayer2D(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, dropout=0, pool=False):
        super(ConvLayer2D, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.kernel = kernel
        self.downConv = weight_norm(nn.Conv2d(in_channels=c_in,
                                              out_channels=c_in,
                                              kernel_size=(kernel, 1) if pool else (1, 1),
                                              stride=(2, 1) if pool else (1, 1),
                                              padding=((kernel - 1) // 2, 0) if pool else(0, 0))
                                    )
        self.activation1 = nn.GELU()
        self.actConv = weight_norm(nn.Conv2d(in_channels=c_in,
                                             out_channels=c_out,
                                             padding=((kernel - 1) // 2, 0),
                                             kernel_size=(kernel, 1))
                                   )
        self.activation2 = nn.GELU()
        self.sampleConv = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=1) if c_in != c_out else None
        self.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)) if pool else None

    def forward(self, x):
        y = x.clone()
        if self.sampleConv is not None:
            y = self.sampleConv(y)
        if self.pool is not None:
            y = self.pool(y)
        x = self.dropout(self.downConv(x))
        x = self.activation1(x)
        x = self.dropout(self.actConv(x))
        x = self.activation2(x)
        x = x + y
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, dropout=0):
        super(AttentionLayer, self).__init__()

        self.inner_attention = AttentionBlock(dropout)
        self.query_projection = weight_norm(nn.Conv2d(in_channels=d_model,
                                                      out_channels=d_model,
                                                      kernel_size=(1, 1))
                                            )
        self.key_projection = weight_norm(nn.Conv2d(in_channels=d_model,
                                                    out_channels=d_model,
                                                    kernel_size=(1, 1))
                                          )
        self.value_projection = weight_norm(nn.Conv2d(in_channels=d_model,
                                                      out_channels=d_model,
                                                      kernel_size=(1, 1))
                                            )
        self.out_projection = weight_norm(nn.Conv2d(in_channels=d_model,
                                                    out_channels=d_model,
                                                    kernel_size=(1, 1))
                                          )

    def forward(self, queries):
        B, D, L, V = queries.shape
        keys = queries.clone()
        values = queries.clone()

        intial_queries = queries.clone()

        queries = self.query_projection(queries).permute(0, 2, 3, 1).view(B, L, V, D)
        keys = self.key_projection(keys).permute(0, 2, 3, 1).view(B, L, V, D)
        values = self.value_projection(values).permute(0, 2, 3, 1).view(B, L, V, D)

        out = self.inner_attention(
            queries,
            keys,
            values
        )

        out = self.out_projection(out.permute(0, 3, 1, 2))

        return intial_queries + out


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, ICOM=False, pool=False, dropout=0):
        super(ConvBlock, self).__init__()
        if ICOM:
            self.conv = nn.Sequential(
                AttentionLayer(c_in, dropout),
                ConvLayer2D(c_in, c_out, kernel, dropout, pool=pool)
            )
        else:
            self.conv = nn.Sequential(
                ConvLayer2D(c_in, c_in, kernel, dropout),
                ConvLayer2D(c_in, c_out, kernel, dropout)
            )

    def forward(self, x):
        x_uni = self.conv(x)  # B D L V
        return x_uni
