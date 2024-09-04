# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : ours.py
# Time       ：30/4/2024 5:34 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：其实dataframe返回的是(item_id, time);  forward 多 DNN concat short, 1e-3;
"""
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence
from stamp import Dnn
from utils import get_last_visit
from torch.nn import MultiheadAttention
import time
###### config
class Args:
    dataset_name = 'video'
    model_name = 'our'
    model_path= '/home/czhaobo/pom/ckps/'
    log_path = '/home/czhaobo/pom/logs/'
    gpu_used = True
    batch_size = 512 # 1-24/256没差别; 256, 512
    lr = 0.002 # 0.003

    weight_decay = 5e-4 # 是否有
    maxlen = 50
    hidden_units = 64
    num_blocks = 1
    num_epochs = 100
    num_heads = 2 #
    dropout_rate = 0.1
    reg = False
    cxt_size = 6
    use_res = True
    log_freq = 100

opt = Args()

###### Model


class Dice(nn.Module):
    r"""Dice activation function

    .. math::
        f(s)=p(s) \cdot s+(1-p(s)) \cdot \alpha s

    .. math::
        p(s)=\frac{1} {1 + e^{-\frac{s-E[s]} {\sqrt {Var[s] + \epsilon}}}}
    """

    def __init__(self, emb_size):
        super(Dice, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.alpha = torch.zeros((emb_size,))

    def forward(self, score):
        self.alpha = self.alpha.to(score.device)
        score_p = self.sigmoid(score)

        return self.alpha * (1 - score_p) * score + score_p * score



class NavieLSTM(nn.Module):
    def __init__(self,input_sz,hidden_sz,batch_first=True):
        super(NavieLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.lstm = torch.nn.LSTM(input_sz,hidden_sz,num_layers=2,batch_first=batch_first)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self,x):
        out,_ = self.lstm(x)
        return out


class CustomLSTMCell(nn.Module):
    """
    实现Cell比较好，这样就不需要管Seq_len了
    """
    def __init__(self, input_sz, hidden_sz):
        super(CustomLSTMCell, self).__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.register_parameter('weight_1',self.W) # rename, no sense
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.register_parameter('weight_2',self.U)
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.register_parameter('bias_1',self.bias)
        # new gate
        self.W_input = nn.Parameter(torch.Tensor(input_sz,hidden_sz*2))
        self.register_parameter('weight_3',self.W_input)
        self.W_delta = nn.Parameter(torch.clamp(nn.Parameter(torch.Tensor(hidden_sz, hidden_sz*2)),max=0).data) # 这里不确定参数限制是否正确
        self.register_parameter('weight_4',self.W_delta) # 需要完全注册，不然self.to(device)不起作用
        self.W_output = nn.Parameter(torch.Tensor(hidden_sz,hidden_sz))
        self.register_parameter('weight_5',self.W_output)
        self.bias_delta = nn.Parameter(torch.Tensor(hidden_sz*2))
        self.register_parameter('bias_2',self.bias_delta)


    def forward(self, inputs_t, hc_t, delta_t):
        x_t = inputs_t # bs, input_sz
        h_t, c_t = hc_t # bs, hidden_sz
        delta_t = delta_t.view(x_t.size(0),-1).repeat(1,h_t.size(1)).float() # bs[tensor] -> bs, hidden_sz
        HS = self.hidden_size
        gates = x_t @ self.W + h_t @ self.U + self.bias
        gate_short = x_t @ self.W_input[:,:HS] + torch.sigmoid(delta_t @ self.W_delta[:,:HS]) + self.bias_delta[:HS]
        gate_long = x_t @ self.W_input[:,HS:] + torch.sigmoid(delta_t @ self.W_delta[:,HS:]) + self.bias_delta[HS:]
        i_t, f_t, g_t, o_t, short, long = (
            torch.sigmoid(gates[:, :HS]),  # input
            torch.sigmoid(gates[:, HS:HS * 2]),  # forget
            torch.tanh(gates[:, HS * 2:HS * 3]),
            torch.sigmoid(gates[:, HS * 3:] + delta_t @ self.W_output),  # output
            torch.sigmoid(gate_short),
            torch.sigmoid(gate_long),
        )
        c_t_short = torch.mul(f_t, c_t) + torch.mul(torch.mul(i_t, g_t), gate_short)
        c_t = torch.mul(f_t, c_t) + torch.mul(torch.mul(i_t, g_t), gate_long)
        h_t = torch.mul(o_t, torch.tanh(c_t_short))
        return h_t, c_t


class DynamicLSTM(nn.Module):
    """这里需要深刻了解这些概念的含义"""
    def __init__(self, input_sz, hidden_sz, lstm_type="Custom"):
        super(DynamicLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz

        if lstm_type == "Custom":
            self.rnn = CustomLSTMCell(input_sz, hidden_sz) # 这里的hidden_sz要注意
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, deltas, init_states=None):
        """
        :param inputs: bs, seq_len,input_size
        :param deltas: bs, seq_len
        :return:
        """
        if not isinstance(inputs, PackedSequence) or not isinstance(deltas,PackedSequence):
            raise NotImplementedError("DynamicLSTM only supports packed input and deltas")

        inputs, batch_sizes, sorted_indices, unsorted_indices = inputs
        deltas, _, _, _ = deltas
        max_batch_size = int(batch_sizes[0])
        # initial param
        if init_states is None:
            h, c = (torch.zeros(max_batch_size, self.hidden_sz,dtype=inputs.dtype,device=inputs.device),
                    torch.zeros(max_batch_size, self.hidden_sz,dtype=inputs.dtype,device=inputs.device))
        elif init_states == "Uniform":
            h, c = init_states
        # bs, hidden_sz, same with lstm[last]
        outputs_h = torch.zeros(inputs.size(0),self.hidden_sz,dtype=inputs.dtype,device=inputs.device)
        outputs_c = torch.zeros(inputs.size(0),self.hidden_sz,dtype=inputs.dtype,device=inputs.device)
        begin = 0
        for batch in batch_sizes: # 这里的bs已经是按照时间步进行划分了
            new_h, new_c = self.rnn(
                inputs[begin:begin+batch], # 每个batch内都是相同大小的参数
                (h[0:batch],c[0:batch]), # 这里h,c是max_batch的参数，no need new parameter
                deltas[begin:begin+batch]
            )
            outputs_h[begin:begin+batch], outputs_c[begin:begin+batch] = new_h, new_c
            h, c = new_h, new_c
            begin += batch
        return PackedSequence(outputs_h,batch_sizes,sorted_indices,unsorted_indices) # c可以暂时不要了,这里也是每一个时刻的





class FuncExtractor(nn.Module): # 设计功能性提取
    def __init__(self, input_size, use_negsampling=False, use_graph=False, use_delta=False):
        super(FuncExtractor, self).__init__()
        self.use_neg = use_negsampling
        self.use_graph = use_graph
        self.use_delta = use_delta
        self.lstm = DynamicLSTM(input_sz=input_size, hidden_sz=input_size, lstm_type="Custom")

        # self.lstm2 = DynamicLSTM(input_sz=input_size, hidden_sz=input_size, lstm_type="Custom")


        if self.use_neg:
            self.auxiliary_net = Dnn(input_size*2,[128,64,1],'sigmoid')

    def forward(self, keys, mask, pos_keys=None, neg_keys=None , delta_f=None, graph_emb=None):
        # 这里的query没有用。可以试试BPR
        batch_size, max_length, dim = keys.size()
        masked_keys = keys # mask.unsqueeze(-1) * keys
        key_length = mask.sum(dim=-1)
        aux_loss = torch.zeros((1,), device=keys.device)

        if self.use_graph and graph_emb is not None:
            # 这里需要把graph emb直接加到LSTM输入中【加到输出难度太大】
            # masked_keys = torch.add(masked_keys,graph_emb)
            masked_keys = masked_keys + graph_emb*mask.unsqueeze(-1) # graph_emb

        masked_deltas = delta_f #* mask.unsqueeze(-1)# torch.masked_select(delta_f, mask.view(-1, 1)).view(-1, max_length)  # 先检查，再转换
        # [B,T,H]
        packed_keys = pack_padded_sequence(masked_keys, lengths=key_length.cpu(), batch_first=True,
                                           enforce_sorted=False)  # lengths参数mask掉长度
        packed_deltas = pack_padded_sequence(masked_deltas, lengths=key_length.cpu(), batch_first=True,
                                           enforce_sorted=False)  # lengths参数mask掉长度
        packed_function = self.lstm(packed_keys, packed_deltas)
        # packed_function = self.lstm2(packed_function, packed_deltas)

        function, _ = pad_packed_sequence(packed_function, batch_first=True, padding_value=0.0, total_length=max_length)
        if self.use_neg and neg_keys is not None:
            # 检查for safe
            masked_neg_keys = neg_keys * mask.unsqueeze(-1)# torch.masked_select(neg_keys, mask.view(-1, 1, 1)).view(-1, max_length, dim)
            masked_pos_keys = pos_keys * mask.unsqueeze(-1)# torch.masked_select(pos_keys, mask.view(-1, 1, 1)).view(-1, max_length, dim)
            aux_loss = self.cal_auxiliary_loss(
                function,  # hidden
                masked_pos_keys,  # pos
                masked_neg_keys,  # neg -> 不是自身就行
                mask)  # [3,3]->[2,2]
        return function, aux_loss  # [B,T,H] ,scalar

    def cal_auxiliary_loss(
            self, states, click_seq, noclick_seq, mask):
        """这里的mask小一号"""
        batch_size, max_seq_length, embedding_size = states.size()

        mask = mask

        click_input = torch.cat([states, click_seq], dim=-1)
        noclick_input = torch.cat([states, noclick_seq], dim=-1)
        embedding_size = embedding_size * 2

        click_p = self.auxiliary_net(
            click_input.view(
                batch_size * max_seq_length, embedding_size)).view(
            batch_size, max_seq_length)[mask > 0].view(-1, 1)
        click_target = torch.ones(
            click_p.size(), dtype=torch.float, device=click_p.device)

        noclick_p = self.auxiliary_net(
            noclick_input.view(
                batch_size * max_seq_length, embedding_size)).view(
            batch_size, max_seq_length)[mask > 0].view(-1, 1)
        noclick_target = torch.zeros(
            noclick_p.size(), dtype=torch.float, device=noclick_p.device)

        loss = F.binary_cross_entropy(
            torch.cat([click_p, noclick_p], dim=0),
            torch.cat([click_target, noclick_target], dim=0))

        return loss

    # def cal_auxiliary_loss(self, states, click_seq, noclick_seq, keys_length):
    #     """
    #     :param states: 隐状态[B, T-1, H]
    #     :param click_seq: 后移正样本
    #     :param noclick_seq: 后移负样本
    #     :param key_length: for mask
    #     :return:
    #     """
    #     # for safe
    #     mask_shape = keys_length > 0
    #     keys_length = keys_length[mask_shape]
    #     if keys_length.shape[0] == 0:
    #         return torch.zeros((1,), device=states.device)
    #     _, max_seq_length, embedding_size = states.size()
    #     states = torch.masked_select(states, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length, embedding_size)
    #     click_seq = torch.masked_select(click_seq, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length,
    #                                                                                embedding_size)
    #     noclick_seq = torch.masked_select(noclick_seq, mask_shape.view(-1, 1, 1)).view(-1, max_seq_length,
    #                                                                                    embedding_size)
    #     batch_size = states.size()[0]
    #     mask = (torch.arange(max_seq_length, device=states.device).repeat(batch_size, 1) < keys_length.view(-1,
    #                                                                                                         1)).float()  # 1表示该位置loss应该回传，否则无
    #     click_input = torch.cat([states, click_seq], dim=-1)
    #     noclick_input = torch.cat([states, noclick_seq], dim=-1)
    #     embedding_size = embedding_size * 2
    #
    #     # 为了方便一次性计算整个Batch内T时刻的情况Tensor(length) -》 只取没有mask的 -》 所有out再次reshape成一个
    #     click_p = self.auxiliary_net(click_input.view(
    #         batch_size * max_seq_length, embedding_size)).view(
    #         batch_size, max_seq_length)[mask > 0].view(-1, 1)
    #     click_target = torch.ones(
    #         click_p.size(), dtype=torch.float, device=click_p.device)
    #
    #     noclick_p = self.auxiliary_net(noclick_input.view(
    #         batch_size * max_seq_length, embedding_size)).view(
    #         batch_size, max_seq_length)[mask > 0].view(-1, 1)
    #     noclick_target = torch.zeros(
    #         noclick_p.size(), dtype=torch.float, device=noclick_p.device)
    #     loss = F.binary_cross_entropy(
    #         torch.cat([click_p, noclick_p], dim=0),
    #         torch.cat([click_target, noclick_target], dim=0)
    #     )
    #
    #     return loss

class MultiLayerAttention(nn.Module):
    def __init__(self, emb_size, num_heads=3, num_layers=3, dropout=0.2):
        super(MultiLayerAttention, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([Attention(emb_size, num_heads) for _ in range(num_layers)])

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size,  emb_size),
            nn.GELU(),
            nn.Linear(emb_size, emb_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, user_emb, long_emb, mask):
        score=None
        for i, layer in enumerate(self.layers):
            user_emb, score = layer(user_emb, long_emb, mask)
            # x = self.dropout(self.norm1(attention + query))
            forward = self.feed_forward(user_emb)
            out = self.dropout(self.norm2(forward + user_emb))
        return out, score



class Attention(nn.Module):
    def __init__(self, emb_size, num_heads=3, dropout=opt.dropout_rate):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_dim = emb_size // num_heads
        assert self.head_dim * num_heads == emb_size, "Embedding size must be divisible by the number of heads"

        self.q_linear = nn.Linear(emb_size, emb_size)
        self.k_linear = nn.Linear(emb_size, emb_size)
        self.v_linear = nn.Linear(emb_size, emb_size)
        self.fc = nn.Linear(emb_size, emb_size)

        self.attention = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_size)

    def forward(self, user_emb, long_emb, mask):
        batch_size, max_seq_length, emb_size = long_emb.size()

        # Linear projections
        Q = self.q_linear(user_emb).view(batch_size, self.num_heads, self.head_dim).unsqueeze(2)  # B, H, 1, D/H
        K = self.k_linear(long_emb).view(batch_size, max_seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, T, D/H
        V = self.v_linear(long_emb).view(batch_size, max_seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, T, D/H

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # B, H, 1, T
        scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)  # B, H, 1, T
        att_weights = self.attention(scores)  # B, H, 1, T

        # Attention output
        att_output = torch.matmul(att_weights, V)  # B, H, 1, D/H
        att_output = att_output.permute(0, 2, 1, 3).contiguous().view(batch_size, 1, self.emb_size)  # B, 1, D

        # Concatenate heads and final linear layer
        output = self.fc(att_output).squeeze(1)  # B, D

        output = self.dropout(output)
        output = self.layer_norm(output + user_emb)

        return output, att_weights

#
#
# class Attention(nn.Module):
#     def __init__(self, emb_size):
#         super(Attention, self).__init__()
#         self.attention = nn.Softmax(dim=-1)
#         # self.linea = nn.Linear(emb_size,emb_size)
#
#     def forward(self, user_emb, long_emb, mask):
#         batch_size, max_seq_length, _ = long_emb.size()
#         # user_emb = self.linea(user_emb)
#
#         user_emb = user_emb.unsqueeze(dim=-1) # B,H,1
#         emb = torch.bmm(long_emb,user_emb)  # [B,T,1]
#         att = torch.squeeze(emb,dim=-1) # [B,T]
#         att = att.masked_fill(mask == 0, -1e9)
#         att_score = self.attention(att).unsqueeze(dim=-1)  # B,T,1
#         score_x = long_emb * att_score  # B ,T ,H
#
#         # double layer
#         # emb = torch.bmm(score_x, user_emb)  # [B,T,1]
#         # att = torch.squeeze(emb,dim=-1) # [B,T]
#         # att = att.masked_fill(mask == 0, -1e9)
#         # att_score = self.attention(att).unsqueeze(dim=-1)  # B,T,1
#         # score_x = score_x * att_score  # B ,T ,H
#
#
#         out = score_x.sum(dim=1)
#         return out, att_score
# class Attention(nn.Module):
#     """
#     层级Attention，汇集序列为一个向量，同样需要Attention mask
#     """
#     def __init__(self,input_sz,hidden_sz):
#         super(Attention,self).__init__()
#         self.w_omega = nn.Parameter(torch.Tensor(input_sz,hidden_sz)) # input_sz, hidden_sz
#         self.u_omega = nn.Parameter(torch.Tensor(hidden_sz,1))
#         nn.init.uniform_(self.w_omega,-0.1,0.1)
#         nn.init.uniform_(self.u_omega,-0.1,0.1)
#         self.attention = nn.Softmax(dim=-1)
#
#     def forward(self, x, mask): # [B,T,H]
#         batch_size,max_seq_length,_ = x.size()
#         u = torch.tanh(torch.matmul(x, self.w_omega)) # [B,T,H]
#         att = torch.squeeze(torch.matmul(u,self.u_omega),dim=-1) # [B,T]
#         # mask = (torch.arange(max_seq_length, device=x.device).repeat(
#         #     batch_size, 1) < keys_length.view(-1, 1)).float()
#         att = att.masked_fill(mask == 0, -1e9)
#         att_score = self.attention(att).unsqueeze(dim=-1) # B,T,H
#         score_x = x * att_score # B ,T ,H
#         out = torch.sum(score_x,dim=1) # B, H
#         return out, att_score



class PrefExtractor(nn.Module): # 设计偏好性提取
    def __init__(self, input_size, use_negsampling=False):
        super(PrefExtractor, self).__init__()
        self.use_neg = use_negsampling
        self.lstm = NavieLSTM(input_sz=input_size,hidden_sz=input_size,batch_first=True)
        if self.use_neg:
            self.auxiliary_net = Dnn(input_size*2,[128,64,1],'sigmoid')

    def forward(self, keys, mask, pos_keys=None, neg_keys=None):
        """
        :param pos_keys: [B,T, H]
        :param keys: [B, T, H]
        :param keys_length: [B]
        :param neg_keys: [B, T, H]
        :return: [B,T,H] [1];
        """
        batch_size, max_length, dim = keys.size()
        aux_loss = torch.zeros((1,),device=keys.device)
        masked_keys = keys # RNN不需要
        # masked_keys = mask.unsqueeze(-1) * keys
        key_length = mask.sum(dim=-1)
        # [B,T,H]
        packed_keys = pack_padded_sequence(masked_keys,lengths=key_length.cpu(),batch_first=True,enforce_sorted=False) # lengths参数mask掉长度
        packed_prefer = self.lstm(packed_keys)
        prefer,_ = pad_packed_sequence(packed_prefer, batch_first=True,padding_value=0.0,total_length=max_length)
        if self.use_neg and neg_keys is not None:
            # 检查for safe
            masked_pos_keys = pos_keys * mask.unsqueeze(-1)
            masked_neg_keys = neg_keys * mask.unsqueeze(-1)
            aux_loss = self.cal_auxiliary_loss(
                prefer[:, 1:, :],  # hidden
                masked_pos_keys[:, :-1, :],  # pos
                masked_neg_keys[:, :-1, :],  # neg -> 不是自身就行
                mask)  # [3,3]->[2,2]

        return prefer, aux_loss  # [B,H] ,scalar

    def cal_auxiliary_loss(
            self, states, click_seq, noclick_seq, mask):
        """这里的mask小一号"""
        batch_size, max_seq_length, embedding_size = states.size()
        mask = mask[:, :-1]

        click_input = torch.cat([states, click_seq], dim=-1)
        noclick_input = torch.cat([states, noclick_seq], dim=-1)
        embedding_size = embedding_size * 2

        click_p = self.auxiliary_net(
            click_input.view(
                batch_size * max_seq_length, embedding_size)).view(
            batch_size, max_seq_length)[mask > 0].view(-1, 1)
        click_target = torch.ones(
            click_p.size(), dtype=torch.float, device=click_p.device)

        noclick_p = self.auxiliary_net(
            noclick_input.view(
                batch_size * max_seq_length, embedding_size)).view(
            batch_size, max_seq_length)[mask > 0].view(-1, 1)
        noclick_target = torch.zeros(
            noclick_p.size(), dtype=torch.float, device=noclick_p.device)

        loss = F.binary_cross_entropy(
            torch.cat([click_p, noclick_p], dim=0),
            torch.cat([click_target, noclick_target], dim=0))

        return loss


class FR(nn.Module):
    def __init__(self, usernum, itemnum, opt, ItemFeatures=None, GraphFeatures=None, cxt_size=None,
                 use_res=False):
        super(FR, self).__init__()
        self.usernum = usernum
        self.itemnum = itemnum
        self.opt = opt
        self.cxt_size = cxt_size
        self.use_res = use_res
        self.ItemFeats = ItemFeatures
        self.GraphFeats = GraphFeatures

        # Embeddings
        self.user_embedding = nn.Embedding(num_embeddings=usernum + 1, embedding_dim=opt.hidden_units)
        self.item_embedding = nn.Embedding(num_embeddings=itemnum + 1, embedding_dim=opt.hidden_units, padding_idx=0)
        self.position_embedding = nn.Embedding(num_embeddings=opt.maxlen, embedding_dim=opt.hidden_units)

        self.seq_feat_emb_layer = nn.Linear(in_features=cxt_size+ self.ItemFeats.shape[1], out_features=opt.hidden_units * (cxt_size-1))

        self.seq_concat_layer = nn.Sequential(nn.Linear(in_features=opt.hidden_units * cxt_size , out_features=opt.hidden_units))
                                              # nn.GELU(), nn.Dropout(opt.dropout_rate),
                                              # nn.Linear(in_features=opt.hidden_units*3, out_features=opt.hidden_units))

        # Item and context features
        if ItemFeatures is not None:
            # self.ItemFeats = nn.Embedding(num_embeddings=itemnum + 1, embedding_dim=args.hidden_units, padding_idx=0)
            self.ItemFeats = nn.Parameter(torch.tensor(ItemFeatures, dtype=torch.float32), requires_grad=False)
        if GraphFeatures is not None: # 这里要变成可学习的
            self.GraphFeats = nn.Parameter(torch.tensor(GraphFeatures, dtype=torch.float32), requires_grad=False)#不同于

        # Blocks
        # for seq_attn
        # blocks = nn.TransformerEncoderLayer(d_model=opt.hidden_units, nhead=opt.num_heads, batch_first=True, dropout=opt.dropout_rate)
        # self.multi_blocks = nn.TransformerEncoder(blocks, num_layers=opt.num_blocks)

        self.delta_proj = nn.Sequential(nn.Linear(cxt_size, 1), nn.Tanh())
        # self.graph_linear = nn.Sequential(nn.Linear(opt.hidden_units, 2*opt.hidden_units),nn.GELU(),nn.Linear(2*opt.hidden_units, opt.hidden_units))
        # self.decompose1 = nn.Sequential(nn.Linear(opt.hidden_units, 2*opt.hidden_units),nn.GELU(),nn.Linear(2*opt.hidden_units, opt.hidden_units))
        # self.decompose2 = nn.Sequential(nn.Linear(opt.hidden_units, 2*opt.hidden_units),nn.GELU(),nn.Linear(2*opt.hidden_units, opt.hidden_units))

        self.graph_linear = nn.Sequential(nn.Linear(opt.hidden_units, opt.hidden_units))
        self.decompose1 = nn.Sequential(nn.Linear(opt.hidden_units, opt.hidden_units),nn.ReLU(),nn.Linear(opt.hidden_units, opt.hidden_units))
        self.decompose2 = nn.Sequential(nn.Linear(opt.hidden_units, opt.hidden_units),nn.ReLU(), nn.Linear(opt.hidden_units, opt.hidden_units))
        self.decompose11 = nn.Sequential(nn.Linear(opt.hidden_units, opt.hidden_units))



        self.rat_extrator = FuncExtractor(input_size=opt.hidden_units, use_negsampling=True, use_graph=True, use_delta=True)
        self.pre_extrator = PrefExtractor(input_size=opt.hidden_units, use_negsampling=True)
        # self.attention = Attention(input_sz=opt.hidden_units,hidden_sz=opt.hidden_units)
        # self.attention = Attention(opt.hidden_units)
        # self.attention = MultiLayerAttention(opt.hidden_units, num_heads=opt.num_heads, num_layers=opt.num_blocks)
        # self.attention = nn.ModuleList([
        #     SelfAttentionBlock(opt.hidden_units, opt.num_heads, opt.dropout_rate, use_res) for _ in
        #     range(1)
        # ])
        decoder_layer = nn.TransformerDecoderLayer(opt.hidden_units, opt.num_heads, batch_first=True)
        self.attention = nn.TransformerDecoder(decoder_layer, num_layers=opt.num_blocks)

        self.dropout = nn.Dropout(opt.dropout_rate)
        self.LayerNorm = nn.LayerNorm(opt.hidden_units)

        # 2个
        self.projp2f=  nn.Dropout(opt.dropout_rate)#nn.Sequential(nn.Linear(opt.hidden_units, opt.hidden_units))#, nn.Dropout(opt.dropout_rate),nn.Linear(opt.hidden_units*2, opt.hidden_units))
        self.projf2p = nn.Dropout(opt.dropout_rate)#nn.Sequential(nn.Linear(opt.hidden_units, opt.hidden_units))#, nn.Dropout(opt.dropout_rate),nn.Linear(opt.hidden_units*2, opt.hidden_units))

        # self.dnn = Dnn(4*opt.hidden_units,hidden_units=(256,128,1),activation='relu',use_bn=False,dropout_rate=0.5)
        self.dnn = nn.Sequential(nn.Linear(in_features=3*opt.hidden_units, out_features=2*opt.hidden_units),
                                   nn.GELU(),
                                 nn.Dropout(opt.dropout_rate),
                                   # nn.Linear(in_features=2*opt.hidden_units, out_features=opt.hidden_units),
                                   # nn.GELU(),
                                   # nn.Dropout(opt.dropout_rate),
                                 nn.Linear(in_features=2*opt.hidden_units, out_features=1)
                                 # nn.GELU(),
                                 # nn.Dropout(opt.dropout_rate),
                                 #   nn.Linear(in_features=2*opt.hidden_units, out_features=1)
                                   )
        # self.dnn = nn.Sequential(
        #     # nn.Dropout(opt.dropout_rate),
        #     nn.Linear(in_features=3 * opt.hidden_units, out_features=2 * opt.hidden_units),
        #     # nn.BatchNorm1d(num_features=2 * opt.hidden_units),
        #     Dice(2 * opt.hidden_units),
        #     nn.Dropout(opt.dropout_rate),
        #     nn.Linear(in_features=2* opt.hidden_units, out_features=1),
        #     # nn.BatchNorm1d(num_features= opt.hidden_units),
        #     # Dice(opt.hidden_units),
        #     # nn.Linear(opt.hidden_units, 1)
        # )

        self.dnn2 = nn.Sequential(nn.Linear(in_features=3 * opt.hidden_units, out_features=2 * opt.hidden_units),
                                 nn.GELU(),
                                 nn.Dropout(opt.dropout_rate),
                                 nn.Linear(in_features=2 * opt.hidden_units, out_features=1)  # )*opt.hidden_units),
                                 # nn.GELU(),
                                 # nn.Dropout(opt.dropout_rate),
                                 # nn.Linear(in_features=2 * opt.hidden_units, out_features=2 * opt.hidden_units),
                                 # nn.GELU(),
                                 # nn.Dropout(opt.dropout_rate),
                                 #   nn.Linear(in_features=2*opt.hidden_units, out_features=1)
                                 )


        self.dropout_user = nn.Sequential(nn.Linear(opt.hidden_units, opt.hidden_units),
                                       nn.GELU(),
                                       nn.Dropout(opt.dropout_rate),
                                          # nn.Linear(2*opt.hidden_units,  opt.hidden_units),
                                       )
        # Prediction layers
        # self.score = nn.Linear(in_features=(cxt_size+2)*opt.hidden_units, out_features=1)

        # self.apply(self._init_weights)
        # self.apply(self.init_weight)


    def orthogonal_loss(self, matrix1, matrix2):
        # 计算两个矩阵的内积
        inner_product = torch.matmul(matrix1.transpose(0, 1), matrix2)
        # 计算正交损失，即内积的平方和减去矩阵的维度
        loss = torch.sum(inner_product ** 2) - matrix1.shape[1]
        return loss

    # def _init_weights(self, module):
    #     """Initialize the weights"""
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=0.01)
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     if isinstance(module, nn.Linear) and module.bias is not None:
    #         module.bias.data.zero_()
    def init_weight(self, module):
        if isinstance(module, nn.Linear):
            # 如果是线性层，则使用Xavier正态分布初始化
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                # 如果存在偏置项，则进行零初始化
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，则进行初始化操作（根据需要定义）
            nn.init.xavier_normal_(module.weight)

    def add_auxiliary_loss(self, aux_loss_p, aux_loss_f, alpha_p=1e-5, alpha_f=1e-5):
        aux_loss = aux_loss_p * alpha_p + aux_loss_f * alpha_f #+ 1e-2 * self.orthogonal_loss(self.decompose1.weight, self.decompose2.weight)
        return aux_loss

    def predict(self, u, input_seq, pos, seq_cxt, pos_cxt, delta_f=None):
        """供eval使用"""
        mask = input_seq!=0
        mask = mask.squeeze(1)
        if delta_f is None:
            delta_f = self.delta_proj(seq_cxt[1].float()).squeeze(1) # B,max_len,1
        # Embedding lookups and initial transformations
        seq_emb = self.item_embedding(input_seq)  # B,max_len, D
        user_tmp = seq_emb.sum(dim=1)
        seq_pos_emb = self.position_embedding(seq_cxt[0])# .unsqueeze(0).repeat(seq_emb.shape[0], 1, 1)  # B,max_len,D
        seq_emb += seq_pos_emb

        seq_emb = self.dropout(seq_emb)
        # seq_emb = self.LayerNorm(seq_emb)

        seq_attr = self.ItemFeats[input_seq]
        seq_cxt = seq_cxt[1].float()
        seq_cxt = self.seq_feat_emb_layer(torch.cat([seq_cxt,seq_attr],dim=-1)) # B,max_len,5D
        concat_emb = torch.cat([seq_cxt, seq_emb], dim=-1)  # B,max_len,6D
        concat_emb = self.seq_concat_layer(concat_emb)
        concat_emb = concat_emb.squeeze(1)# B,max_len,D
        u_emb = self.user_embedding(u)  # B,max_len,D
        # u_emb = self.user_proj(u_emb)
        u_emb = self.dropout_user(u_emb)


        rat_emb, emo_emb = self.decompose1(concat_emb), self.decompose2(concat_emb)
        
        # rat_emb, emo_emb =  emo_emb, rat_emb
        # rat_emb, emo_emb = F.relu(rat_emb), F.relu(emo_emb)

        # target
        pos_emb = self.item_embedding(pos) # B,101, D
        pos_attr = self.ItemFeats[pos] # B,101, 5
        pos_cxt_emb = pos_cxt[1].float()# B, 101, 5
        pos_cxt_emb = self.seq_feat_emb_layer(torch.cat([pos_cxt_emb,pos_attr],dim=-1)) # B, 101, 5D
        pos_emb = torch.cat([pos_cxt_emb, pos_emb], dim=-1) # B, 101, 6D
        pos_emb = self.seq_concat_layer(pos_emb) # B,101,D
        pos_emb_r, pos_emb_p = self.decompose11(pos_emb), self.decompose2(pos_emb) # B,101,D, 可能还需要concat下

        # masked_interest_p, _ = self.pre_extrator(emo_emb, mask) # B，T，H
        # masked_interest_p, att_score = self.attention(u_emb[:,-1,:], emo_emb, mask) # [B,H]

        emo_emb = emo_emb * mask.unsqueeze(-1)
        attn_mask = ~torch.tril(torch.ones((emo_emb.shape[1], emo_emb.shape[1]), dtype=torch.bool, device=concat_emb.device))
        # for submod in self.attention:
        #     _, attn_emo = submod(u_emb, emo_emb, emo_emb, None, attn_mask)
        # masked_interest_p = torch.bmm(attn_emo, emo_emb)  # B,max_len,D
        masked_interest_p = self.attention(u_emb, emo_emb, memory_mask=attn_mask)


        graph_emb = self.graph_linear(self.GraphFeats[input_seq]) # B,max_len,D
        graph_emb = graph_emb.squeeze(1)

        masked_interest_f, aux_loss_r = self.rat_extrator(rat_emb, mask, None, None, delta_f, graph_emb) #B,max_len,D
        masked_interest_f *= mask.unsqueeze(-1) # B,max_len,D

        # masked_interest_f_short = get_last_visit(masked_interest_f, mask) # B,D
        # masked_interest_f_long = masked_interest_f.sum(dim=1)/mask.sum(dim=-1).unsqueeze(dim=-1)# B,D
        # masked_interest_f = torch.cat([masked_interest_f_long,masked_interest_f_short],dim=-1) #masked_interest_f_short + masked_interest_f_long

        masked_interest_f, masked_interest_p = get_last_visit(masked_interest_f, mask), get_last_visit(masked_interest_p, mask) # B,D


        masked_interest_f = masked_interest_f.unsqueeze(1).repeat(1, pos_emb.shape[1], 1) # B,101,D
        masked_interest_p = masked_interest_p.unsqueeze(1).repeat(1,pos_emb.shape[1],1)
        u_emb = u_emb[:,-1,:].unsqueeze(1).repeat(1, pos_emb.shape[1],1)
        masked_interest_p = masked_interest_p  + u_emb
        # pos_emb_p = torch.cat([pos_emb_p, masked_interest_p+ self.projf2p(masked_interest_f)], dim=-1) # B,101,4D
        # pos_emb_r = torch.cat([pos_emb_r, self.projp2f(masked_interest_p)+ masked_interest_f], dim=-1)
        pos_emb_p = torch.cat([pos_emb_r, masked_interest_f,masked_interest_p], dim=-1)
        # pos_emb_r = torch.cat([pos_emb_r, masked_interest_f], dim=-1)
        pos_logit = self.dnn(pos_emb_p) #+ self.dnn(pos_emb_p) #+ self.dnn2(pos_emb_p) + self.dnn2(pos_emb_r)# 这里注意别搞反了

        output = {
            'logit': pos_logit.squeeze(-1),
            'rational': masked_interest_f[:,0,:], # B, D
            'emotional': masked_interest_p[:,0,:], # B, D
            'item_rational': pos_emb[:, 0, :] # B, D
        }
        return output# pos_logit.squeeze(-1)

    # def convert_pad(self, sequences):
    #     """前向padding转为后向"""
    #     # 计算每个序列的实际长度
    #     lengths = (sequences != 0).sum(dim=1)
    #
    #     # 创建一个新的空tensor用于后向padding
    #     max_length = sequences.size(1)
    #     padded_sequences = torch.zeros_like(sequences)
    #
    #     # 生成范围内的索引
    #     idx = torch.arange(max_length).expand(len(lengths), max_length).to(sequences.device)
    #
    #     # 使用长度生成mask，当index小于实际长度时，mask为True
    #     mask = idx < lengths.unsqueeze(1)
    #
    #     # 应用mask来从原始序列复制数据到新的后向padded序列
    #     padded_sequences[mask] = sequences[sequences != 0].flatten()
    #     return padded_sequences.to(sequences.device)

    def forward(self, u, input_seq, pos, neg, seq_cxt, pos_cxt, neg_cxt, delta_f=None):
        """
        获取emb; 获取短期emb，获取长期emb，然后concat进行学习。这里的数据集处理不一致，因为其做sub sequence截断，而CACAR使用的是偏移序列。; 这里不分
        """
        mask = input_seq!=0
        if delta_f is None:
            delta_f = self.delta_proj(seq_cxt[1]).squeeze()
        # print(input_seq.shape, delta_f.shape)
        # Embedding lookups and initial transformations
        seq_emb = self.item_embedding(input_seq)  # B,max_len, D
        seq_pos_emb = self.position_embedding(seq_cxt[0]).unsqueeze(0).repeat(seq_emb.shape[0], 1, 1)  # B,max_len,D
        seq_emb += seq_pos_emb

        seq_emb = self.dropout(seq_emb)


        # seq_emb = self.dropout(seq_emb)
        # seq_emb = self.LayerNorm(seq_emb)

        seq_attr = self.ItemFeats[input_seq]
        seq_cxt = seq_cxt[1]
        seq_cxt = self.seq_feat_emb_layer(torch.cat([seq_cxt,seq_attr],dim=-1)) # B,max_len,5D
        concat_emb = torch.cat([seq_cxt, seq_emb], dim=-1)  # B,max_len,6D
        concat_emb = self.seq_concat_layer(concat_emb) # B,max_len,D



        u_emb = self.user_embedding(u)
        u_emb = self.dropout_user(u_emb)

        # u_emb = u_emb[:,-1,:]  # B,D
        # u_emb = self.user_proj(u_emb)


        # attn_mask = ~torch.tril(torch.ones((concat_emb.shape[1], concat_emb.shape[1]), dtype=torch.bool, device=concat_emb.device))
        # concat_emb = self.multi_blocks(concat_emb, src_key_padding_mask=None, mask=attn_mask) # B,max_len,D

        rat_emb, emo_emb = self.decompose1(concat_emb), self.decompose2(concat_emb)
        # rat_emb, emo_emb = F.relu(rat_emb), F.relu(emo_emb)

        # target
        pos_emb = self.item_embedding(pos) # B, max_len, D
        pos_attr = self.ItemFeats[pos] # B, max_len, 5
        pos_cxt_emb = pos_cxt[1].float() # B, max_len, 5
        pos_cxt_emb = self.seq_feat_emb_layer(torch.cat([pos_cxt_emb,pos_attr],dim=-1)) # B, max_len, 5D
        pos_emb = torch.cat([pos_cxt_emb, pos_emb], dim=-1) # B, max_len, 6D
        neg_emb = self.item_embedding(neg)
        neg_attr = self.ItemFeats[neg]
        neg_cxt_emb = neg_cxt[1].float()
        neg_cxt_emb = self.seq_feat_emb_layer(torch.cat([neg_cxt_emb,neg_attr],dim=-1))
        neg_emb = torch.cat([neg_cxt_emb, neg_emb], dim=-1)  # B, max_len, 6D
        pos_emb, neg_emb = self.seq_concat_layer(pos_emb), self.seq_concat_layer(neg_emb)
        pos_emb, neg_emb = pos_emb, neg_emb
        pos_emb_r, pos_emb_p = self.decompose11(pos_emb), self.decompose2(pos_emb) # 这样保证完全没有关系
        neg_emb_r, neg_emb_p = self.decompose11(neg_emb), self.decompose2(neg_emb)

        # masked_interest_p, aux_loss_p = self.pre_extrator(emo_emb, mask,pos_emb_p, neg_emb_p)
        # masked_interest_p, att_score = self.attention(emo_emb, mask) # [B,H], 这里没啥问题啊


        # masked_interest_p, _ = self.attention(u_emb, emo_emb, mask) # [B,H], 这里没啥问题啊

        # masked_interest_p = self.attention(u_emb, emo_emb, emo_emb, mask)

        emo_emb = emo_emb * mask.unsqueeze(-1)
        attn_mask = ~torch.tril(torch.ones((emo_emb.shape[1], emo_emb.shape[1]), dtype=torch.bool, device=concat_emb.device))
        # for submod in self.attention:
        #     _, attn_emo = submod(u_emb, emo_emb, emo_emb, None, attn_mask)
        # masked_interest_p = torch.bmm(attn_emo, emo_emb)  # B,max_len,D
        masked_interest_p = self.attention(u_emb, emo_emb, memory_mask=attn_mask) + u_emb



        graph_emb = self.graph_linear(self.GraphFeats[input_seq])

        masked_interest_f, aux_loss_r = self.rat_extrator(rat_emb, mask, pos_emb_r, neg_emb_r, delta_f, graph_emb) # [隐式] [B,T,H]
        masked_interest_f *= mask.unsqueeze(-1) # B,max_len,D
        # masked_interest_f_short = get_last_visit(masked_interest_f, mask)
        # masked_interest_f_long = masked_interest_f.sum(dim=1)/mask.sum(dim=-1).unsqueeze(dim=-1)# B,D
        # masked_interest_f = torch.cat([masked_interest_f_long,masked_interest_f_short],dim=-1) #masked_interest_f_short + masked_interest_f_long

        # pos_emb_r = get_last_visit(pos_emb_r, mask)
        # pos_emb_p = get_last_visit(pos_emb_p, mask)
        # neg_emb_r = get_last_visit(neg_emb_r, mask)
        # neg_emb_p = get_last_visit(neg_emb_p, mask)
        aux_loss = self.add_auxiliary_loss(0, aux_loss_r)

        # pos_emb_r, neg_emb_r = torch.cat([pos_emb_r, self.projp2f(masked_interest_p)+ masked_interest_f], dim=-1), torch.cat([neg_emb_r, self.projp2f(masked_interest_p)+ masked_interest_f], dim=-1)
        #
        # pos_emb_p, neg_emb_p = torch.cat([pos_emb_p, masked_interest_p+ self.projf2p(masked_interest_f)], dim=-1), torch.cat([neg_emb_p, masked_interest_p+self.projf2p(masked_interest_f)], dim=-1)
        pos_emb_r, neg_emb_r = torch.cat([pos_emb_r, masked_interest_f,masked_interest_p], dim=-1), torch.cat([neg_emb_r, masked_interest_f,masked_interest_p], dim=-1)
        # pos_emb_p, neg_emb_p = torch.cat([, ], dim=-1), torch.cat([, ], dim=-1)
        pos_logit = self.dnn(pos_emb_r) #+ self.dnn(pos_emb_r)
        neg_logit = self.dnn(neg_emb_r) #+ self.dnn(neg_emb_r)

        out = torch.cat([pos_logit, neg_logit], dim=-1).squeeze(dim=1)
        # print("A", out.shape)
        return out, aux_loss
        # return out.unsqueeze(dim=1), aux_loss




class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, use_res):
        super(SelfAttentionBlock, self).__init__()
        self.multihead_attention = MultiheadAttention(hidden_units, num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_res = use_res
        # Add other necessary layers, like normalization, feedforward, etc.

    def forward(self, query, key, value, mask, attn_mask=None):
        # Implement the forward pass
        # Remember to handle residuals and dropout if necessary
        x, attn = self.multihead_attention(query, key, value, key_padding_mask=mask, attn_mask=attn_mask)
        if self.use_res:
            x = x + query
        return x, attn





# class CrossAttentionDecoderLayer(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
#         super(CrossAttentionDecoderLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)
#
#     def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
#         tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
#                               key_padding_mask=tgt_key_padding_mask)[0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)
#
#         tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask,
#                                key_padding_mask=memory_key_padding_mask)[0]
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)
#
#         tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
#         tgt = tgt + self.dropout3(tgt2)
#         tgt = self.norm3(tgt)
#
#         return tgt

if __name__ == '__main__':
    ItemFeatures = np.random.randn(10,6) # 10个商品
    GraphFeatures = np.random.randn(10,90) # 10个商品
    usernum = 3
    itemnum = 10
    model = FR(usernum, itemnum, opt, ItemFeatures=ItemFeatures, GraphFeatures=GraphFeatures, cxt_size=6,
                 use_res=False)
    print("Init success! ")

    u = torch.tensor([1,2,3])
    input_seq = torch.tensor([[1,2,3],[2,3,4],[3,4,5]])
    pos = torch.tensor([[1,2,3],[2,3,4],[3,4,5]])
    neg = torch.tensor([[0,1,2],[1,2,3],[4,5,6]])
    seq_cxt = [torch.arange(3),torch.tensor([[1,2,3],[2,3,4],[3,4,5]])] # 这里3对应max_len
    pos_cxt = [torch.arange(3),torch.tensor([[1,2,3],[2,3,4],[3,4,5]])]
    neg_cxt = [torch.arange(3),torch.tensor([[1,2,3],[2,3,4],[3,4,5]])]

    delta_f = torch.tensor([[0,1,2],[1,2,3],[4,5,6]])

    out = model(u, input_seq, pos, neg, seq_cxt, pos_cxt, neg_cxt, delta_f)
    print(out[0].shape)
    out = model.predict(u, input_seq, pos, seq_cxt, pos_cxt, delta_f)
    print(out.shape)
