# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : caser.py
# Time       ：30/4/2024 5:34 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：整个序列；1
"""


import torch
import numpy as np
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.nn import functional as F
from utils import get_last_visit


###### config
class Args:
    dataset_name = 'steam'
    model_name = 'caser'
    model_path= '/home/czhaobo/HyperHealth/src/pom/ckps/'
    log_path = '/home/czhaobo/HyperHealth/src/pom/logs/'
    gpu_used = True
    batch_size = 128
    lr = 0.001
    weight_decay = 5e-4
    maxlen = 50
    hidden_units = 90
    num_blocks = 3
    num_epochs = 100
    num_heads = 3 #
    dropout_rate = 0.5 #2
    reg = False
    cxt_size = 6
    use_res = True
    log_freq = 200

opt = Args()

###### Model
class CASER(nn.Module):
    def __init__(self, usernum, itemnum, opt, ItemFeatures=None, UserFeatures=None, cxt_size=None,
                 use_res=False):
        super(CASER, self).__init__()
        self.usernum = usernum
        self.itemnum = itemnum
        self.opt = opt
        self.cxt_size = cxt_size
        self.use_res = use_res
        self.ItemFeats = ItemFeatures

        # Embeddings
        self.user_embedding = nn.Embedding(num_embeddings=usernum + 1, embedding_dim=opt.hidden_units)
        self.item_embedding = nn.Embedding(num_embeddings=itemnum + 1, embedding_dim=opt.hidden_units, padding_idx=0)
        self.position_embedding = nn.Embedding(num_embeddings=opt.maxlen, embedding_dim=opt.hidden_units)

        self.seq_feat_emb_layer = nn.Linear(in_features=cxt_size+self.ItemFeats.shape[-1], out_features=opt.hidden_units * (cxt_size-1))

        self.seq_concat_layer = nn.Linear(in_features=opt.hidden_units * cxt_size, out_features=opt.hidden_units)
        self.cxt_concat_layer = nn.Linear(in_features=opt.hidden_units* (cxt_size-1), out_features=opt.hidden_units)

        # Item and context features
        if ItemFeatures is not None:
            # self.ItemFeats = nn.Embedding(num_embeddings=itemnum + 1, embedding_dim=args.hidden_units, padding_idx=0)
            self.ItemFeats = nn.Parameter(torch.tensor(ItemFeatures, dtype=torch.float32), requires_grad=False)


        # Blocks
        # for seq_attn
        self.max_len = opt.maxlen
        self.n_v = 2
        self.n_h = 2
        # vertical conv layer
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=self.n_v, kernel_size=(self.max_len, 1)) # maxlen一定要大于等于每个batch的实际长度
        # horizontal conv layer
        lengths = [i + 1 for i in range(self.max_len)]
        self.conv_h = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.n_h, kernel_size=(i, opt.hidden_units)) for i in lengths
        ])
        # fully-connected layer
        self.fc1_dim_v = self.n_v * opt.hidden_units
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, opt.hidden_units)
        self.fc2 = nn.Linear(opt.hidden_units*2, opt.hidden_units)
        self.dropout = nn.Dropout(0.5)
        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()


        # Prediction layers
        # self.score = nn.Linear(in_features=2*opt.hidden_units, out_features=1)
        self.score = nn.Sequential(nn.Linear(in_features=2*opt.hidden_units, out_features=2*opt.hidden_units),
                                   nn.LeakyReLU(),
                                   nn.Dropout(opt.dropout_rate),
                                   # nn.Linear(in_features=2*opt.hidden_units, out_features=opt.hidden_units),
                                   # nn.LeakyReLU(),
                                   # nn.Dropout(opt.dropout_rate),
                                   nn.Linear(in_features=2*opt.hidden_units, out_features=1)
                                   )

    def predict(self, u, input_seq, pos, seq_cxt, pos_cxt):
        """供eval使用"""
        mask = input_seq!=0
        mask = mask.squeeze(1)
        # Embedding lookups and initial transformations
        seq_emb = self.item_embedding(input_seq)  # B,max_len, D
        seq_pos_emb = self.position_embedding(seq_cxt[0])# .unsqueeze(0).repeat(seq_emb.shape[0], 1, 1)  # B,max_len,D
        seq_emb += seq_pos_emb
        # seq_cxt = self.ItemFeats[seq_cxt[1]]
        seq_attr = self.ItemFeats[input_seq]
        seq_cxt = seq_cxt[1].float()
        seq_cxt = self.seq_feat_emb_layer(torch.cat([seq_cxt,seq_attr],dim=-1)) # B,max_len,5D
        concat_emb = torch.cat([seq_cxt, seq_emb], dim=-1)  # B,max_len,6D
        concat_emb = self.seq_concat_layer(concat_emb) # B,max_len,1D
        concat_emb = concat_emb.squeeze(1)
        concat_emb *= mask.unsqueeze(-1)
        u_emb = self.user_embedding(u) # B,max_len, D

        # seq
        # Convolutional Layers
        out, out_h, out_v = None, None, None
        keys_emb = concat_emb.unsqueeze(dim=1)

        if self.n_v:
            out_v = self.conv_v(keys_emb) # B,n_v,1,H
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect. B,n_v*H
        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(keys_emb).squeeze(3))  # B,n_h,H
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # B,n_h*H
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect,B,T*n_h
        out = torch.cat([out_v, out_h], 1) # B,T*(n_v+n_h)
        # apply dropout
        out = self.dropout(out)
        # fully-connected layer
        z = self.ac_fc(self.fc1(out)) # B,H
        u_emb = u_emb[:,-1,:] # B,D
        x = torch.cat([z, u_emb], 1) # B,2*H(user_emb+input_sz)
        seq_output = self.ac_fc(self.fc2(x)) # B,H

        # target
        pos_emb = self.item_embedding(pos) # B, 101, D
        # pos_cxt_emb = self.ItemFeats[pos_cxt[1]] # B, 101, 5
        pos_attr = self.ItemFeats[pos]
        pos_cxt_emb = pos_cxt[1].float()
        seq_output = seq_output.unsqueeze(1).repeat(1,pos_emb.shape[1],1)# B,101,D

        pos_cxt_emb = self.seq_feat_emb_layer(torch.cat([pos_cxt_emb,pos_attr],dim=-1)) # B, max_len, 5D
        pos_emb = torch.cat([pos_emb, pos_cxt_emb], dim=-1) # B, max_len, 6D
        pos = self.seq_concat_layer(pos_emb)
        pos_emb = torch.cat([pos, seq_output], dim=-1)
        pos_logit = self.score(pos_emb)
        return pos_logit.squeeze(-1)



    def forward(self, u, input_seq, pos, neg, seq_cxt, pos_cxt, neg_cxt):
        """
        获取emb; 获取短期emb，获取长期emb，然后concat进行学习。这里的数据集处理不一致，因为其做sub sequence截断，而CACAR使用的是偏移序列。; 这里不分
        """
        mask = input_seq!=0

        # Embedding lookups and initial transformations
        seq_emb = self.item_embedding(input_seq)  # B,max_len, D
        seq_pos_emb = self.position_embedding(seq_cxt[0]).unsqueeze(0).repeat(seq_emb.shape[0], 1, 1)  # B,max_len,D
        seq_emb += seq_pos_emb
        # seq_cxt = self.ItemFeats[seq_cxt[1]]
        seq_attr = self.ItemFeats[input_seq]
        seq_cxt = seq_cxt[1]
        seq_cxt = self.seq_feat_emb_layer(torch.cat([seq_cxt,seq_attr],dim=-1)) # B,max_len,5D
        concat_emb = torch.cat([seq_cxt, seq_emb], dim=-1)  # B,max_len,6D
        concat_emb = self.seq_concat_layer(concat_emb) # B,max_len,1D
        concat_emb *= mask.unsqueeze(-1)
        u = u[:,0]
        u_emb = self.user_embedding(u) # B, D

        # seq
        # Convolutional Layers
        out, out_h, out_v = None, None, None
        keys_emb = concat_emb.unsqueeze(dim=1)

        if self.n_v:
            out_v = self.conv_v(keys_emb) # B,n_v,1,H
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect. B,n_v*H
        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(keys_emb).squeeze(3))  # B,n_h,H
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # B,n_h*H
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect,B,T*n_h
        out = torch.cat([out_v, out_h], 1) # B,T*(n_v+n_h)
        # apply dropout
        out = self.dropout(out)
        # fully-connected layer
        z = self.ac_fc(self.fc1(out)) # B,H

        x = torch.cat([z, u_emb], 1) # B,2*H(user_emb+input_sz)
        seq_output = self.ac_fc(self.fc2(x)) # B,H

        # target
        pos_emb = self.item_embedding(pos) # B, max_len, D
        # pos_cxt_emb = self.ItemFeats[pos_cxt[1]] # B, max_len, 5
        pos_fea = self.ItemFeats[pos]
        pos_cxt_emb = pos_cxt[1]
        pos_cxt_emb = self.seq_feat_emb_layer(torch.cat([pos_cxt_emb,pos_fea],dim=-1)) # B, max_len, 5D
        pos_emb = torch.cat([pos_emb, pos_cxt_emb], dim=-1) # B, max_len, 6D
        neg_emb = self.item_embedding(neg)
        # neg_cxt_emb = self.ItemFeats[neg_cxt[1]]
        neg_fea = self.ItemFeats[neg]
        neg_cxt_emb = neg_cxt[1]
        neg_cxt_emb = self.seq_feat_emb_layer(torch.cat([neg_cxt_emb,neg_fea],dim=-1))
        neg_emb = torch.cat([neg_emb, neg_cxt_emb], dim=-1)  # B, max_len, 6D

        pos_emb, neg_emb = self.seq_concat_layer(pos_emb), self.seq_concat_layer(neg_emb)
        pos, neg = pos_emb[:,-1,:].unsqueeze(dim=1), neg_emb[:,-1,:].unsqueeze(dim=1)

        pos_emb, neg_emb = torch.cat([pos, seq_output.unsqueeze(dim=1)], dim=-1), torch.cat([neg, seq_output.unsqueeze(dim=1)], dim=-1)
        pos_logit = self.score(pos_emb)
        neg_logit = self.score(neg_emb)

        out = torch.cat([pos_logit, neg_logit], dim=-1)
        return out,0



if __name__ == '__main__':
    ItemFeatures = np.random.randn(10,6) # 10个商品
    usernum = 6
    itemnum = 10
    model = CASER(usernum, itemnum, opt, ItemFeatures=ItemFeatures, UserFeatures=None, cxt_size=6,
                 use_res=False)
    print("Init success! ")

    u = torch.tensor([1,2,3,4,5])
    input_seq = torch.tensor([[1,2,3],[2,3,4],[3,4,5],[2,3,4],[3,4,5]])
    pos = torch.tensor([[1,2,3],[2,3,4],[3,4,5],[2,3,4],[3,4,5]])
    neg = torch.tensor([[0,1,2],[1,2,3],[4,5,6],[2,3,4],[3,4,5]])
    seq_cxt = [torch.arange(3),torch.tensor([[1,2,3],[2,3,4],[3,4,5],[2,3,4],[3,4,5]])] # 这里3对应max_len
    pos_cxt = [torch.arange(3),torch.tensor([[1,2,3],[2,3,4],[3,4,5],[2,3,4],[3,4,5]])]
    neg_cxt = [torch.arange(3),torch.tensor([[1,2,3],[2,3,4],[3,4,5],[2,3,4],[3,4,5]])]

    out = model(u, input_seq, pos, neg, seq_cxt, pos_cxt, neg_cxt)
    print(out.shape)

    out = model.predict(u, input_seq, pos, seq_cxt, pos_cxt)
    print(out.shape)
