import numpy as np
from torch import nn


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, i_emb_seq, u_emb):
        # self attention
        S = nn.Softmax(dim=-1)((i_emb_seq @ i_emb_seq.T) / np.sqrt(u_emb.shape[0])) @ i_emb_seq
        # attention, which a user's global embedding as query
        return nn.Softmax(dim=-1)((u_emb.unsqueeze(0) @ S.T) / np.sqrt(u_emb.shape[0])) @ i_emb_seq
