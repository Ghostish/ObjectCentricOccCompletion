import torch
from torch import nn as nn
from mmdet3d.ops.sst.sst_ops import get_activation_layer
from mmcv.runner import BaseModule, force_fp32
import math
import copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, abs_pos):
        """
        Arguments:
            abs_pos: Tensor, shape ``[seq_len, batch_size]``

            return ``[seq_len, batch_size, d_model]``
        """
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=abs_pos.device)
            * (-math.log(10000.0) / self.d_model)
        )  # [d_model/2]
        pe = torch.cat(
            [
                torch.sin(abs_pos[..., None] * div_term),
                torch.cos(abs_pos[..., None] * div_term),
            ],
            dim=-1,
        )
        return pe


class SimpleEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        mlp_dropout=0,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(mlp_dropout)
        self.dropout2 = nn.Dropout(mlp_dropout)

        self.activation = get_activation_layer(activation)
        self.fp16_enabled = False

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    @force_fp32(apply_to=("src"))
    def forward(
        self,
        src,
        key_padding_mask=None,
        pos_enc=None,
        attn_mask=None,
    ):
        q = k = self.with_pos_embed(src, pos_enc)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )[
            0
        ]  # [N, d_model]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self,encoder_layer,num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self,src,key_padding_mask=None,pos_enc=None,attn_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output,key_padding_mask,pos_enc,attn_mask)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self,decoder_layer,num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self,tgt,memory,tgt_mask=None,memory_mask=None,
        tgt_key_padding_mask=None,memory_key_padding_mask=None,
        pos_enc=None,
        query_pos_enc=None,):
        output = tgt
        for layer in self.layers:
            output = layer(output,memory,
                           tgt_mask= tgt_mask,memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=memory_key_padding_mask,
                           pos_enc=pos_enc,query_pos_enc=query_pos_enc)
        return output

class SimpleDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        mlp_dropout=0,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout,
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(mlp_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(mlp_dropout)
        self.dropout2 = nn.Dropout(mlp_dropout)
        self.dropout3 = nn.Dropout(mlp_dropout)

        self.activation = get_activation_layer(activation)
        self.fp16_enabled = False

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    @force_fp32(apply_to=("src"))
    def forward(
        self,
        tgt,memory,tgt_mask=None,memory_mask=None,
        tgt_key_padding_mask=None,memory_key_padding_mask=None,
        pos_enc=None,
        query_pos_enc=None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos_enc)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[
            0
        ]  
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt,query_pos_enc), 
            key=self.with_pos_embed(memory,pos_enc), 
            value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])