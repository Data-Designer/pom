# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : utils.py
# Time       ：29/4/2024 7:21 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：several common tools
"""
import math
import torch
# import torchmetrics
import pickle
import gzip
import numpy as np
from sklearn.metrics import roc_auc_score

from collections import defaultdict



def getAUC(score, targetItem=0):
    """计算AUC"""
    tmpans = 0
    count = 0
    for j in range(1, len(score)):  # sample
        if score[targetItem] > score[j]:
            tmpans += 1
        count += 1
    tmpans /= float(count)
    return tmpans


def pearson_corr(A, B):
    mean_A = A.mean(axis=-1, keepdims=True)
    mean_B = B.mean(axis=-1, keepdims=True)

    A_centered = A - mean_A
    B_centered = B - mean_B

    numerator = np.sum(A_centered * B_centered, axis=-1)
    denominator = np.sqrt(np.sum(A_centered ** 2, axis=-1) * np.sum(B_centered ** 2, axis=-1))

    return np.mean(numerator / denominator)



def spearman_rank_correlation(X, Y):
    # Get the ranks for each element in X and Y
    rank_X = np.argsort(np.argsort(X, axis=1), axis=1)
    rank_Y = np.argsort(np.argsort(Y, axis=1), axis=1)

    # Calculate the covariance between rank variables for each sample
    covariance = np.mean((rank_X - rank_X.mean(axis=1, keepdims=True)) *
                         (rank_Y - rank_Y.mean(axis=1, keepdims=True)), axis=1)

    # Calculate the standard deviations of the rank variables for each sample
    std_X = np.std(rank_X, axis=1)
    std_Y = np.std(rank_Y, axis=1)

    # Calculate the Spearman rank correlation coefficient for each sample
    spearman_correlation = covariance / (std_X * std_Y)

    return np.mean(spearman_correlation)
def cosine(A, B):
    cosine_sim = np.sum(A * B, axis=-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1))
    return np.mean(cosine_sim)

def euclidean(A, B):
    return np.mean(np.linalg.norm(A - B, axis=-1))

def getHitRatio(ranklist, rank):
    """首位为0"""
    if ranklist[0] < rank:
        return 1
    # for item in ranklist: #[0]<10 [16,10,4,3,1]
    #     if item == targetItem:
    #         return 1
    return 0


def getNDCG(ranklist, rank):
    # for i in range(len(ranklist)):
    if ranklist[0] < rank:
        return 1 / np.log2(ranklist[0].cpu().numpy() + 2) # math.log(2) / math.log(i+2)
    return 0


def getMRR(ranklist, rank):
    """留一法MAP一样，这和NDCG没区别啊"""
    # score = 0
    # for index,item in enumerate(ranklist,start=1):
    #     if item == rank:
    if ranklist[0] < rank:
        score = 1/(ranklist[0].cpu().numpy()+1)
        return score
    return 0


def cal_metrics(out, ranklists, test=False, y_group=None, rank_items=None):
    output = defaultdict(float)
    # y_group = y_group # {'Attack': [2,4,8]}


    if test:
        # if y_group is not None:
        #     y_correct_ndcg = {key: 0 for key in y_group.keys()}
        #     y_total_ndcg = {key: 0 for key in y_group.keys()}
        #     y_correct_hit = {key: 0 for key in y_group.keys()}
        #     y_total_hit = {key: 0 for key in y_group.keys()}
        #     y_correct_mrr = {key: 0 for key in y_group.keys()}
        #     y_total_mrr = {key: 0 for key in y_group.keys()}

        output['auc'] = 0 # get_metric(out, 'auc')太多了
        for i in range(ranklists.shape[0]): # B,100
            ranklist = ranklists[i]

            # rank_item = rank_items[i]
            # for i in y_group.keys():
            #     # print('AAAAAAAAAA', rank_item[0])
            #     # print(y_group)
            #     if int(rank_item[0]) in y_group[i]: # 这里不大对劲，应该这里放的是推荐的top5个产品
            #         y_total_ndcg[i] += 1
            #         y_total_hit[i] += 1
            #         y_total_mrr[i] += 1
            #         y_correct_ndcg[i] += get_metric(ranklist, 'ndcg', rank=5)
            #         y_correct_hit[i] += get_metric(ranklist, 'hit', rank=5)
            #         y_correct_mrr[i] += get_metric(ranklist, 'mrr', rank=5)


            # print(ranklist, get_metric(ranklist, 'hit', rank=10))
            output['hit_1'] += get_metric(ranklist, 'hit', rank=1)
            output['hit_5'] += get_metric(ranklist, 'hit', rank=5)
            output['hit_10'] += get_metric(ranklist, 'hit', rank=10)
            output['hit_15'] +=  get_metric(ranklist, 'hit', rank=15)
            output['hit_20'] +=  get_metric(ranklist, 'hit', rank=20)
            output['ndcg_1'] += get_metric(ranklist, 'ndcg', rank=1)
            output['ndcg_5'] += get_metric(ranklist, 'ndcg', rank=5)
            output['ndcg_10'] += get_metric(ranklist, 'ndcg', rank=10)
            output['ndcg_15'] +=  get_metric(ranklist, 'ndcg', rank=15)
            output['ndcg_20'] +=  get_metric(ranklist, 'ndcg', rank=20)
            output['mrr_1'] += get_metric(ranklist, 'mrr', rank=1)
            output['mrr_5'] += get_metric(ranklist, 'mrr', rank=5)
            output['mrr_10'] += get_metric(ranklist, 'mrr', rank=10)
            output['mrr_15'] +=  get_metric(ranklist, 'mrr', rank=15)
            output['mrr_20'] +=  get_metric(ranklist, 'mrr', rank=20)
        for i in output.keys():
            output[i] /= ranklists.shape[0]

        # if y_group is not None:
        #     # print(y_correct_ndcg, y_total_ndcg)
        #     for i in y_group.keys():
        #         if y_total_ndcg[i] == 0:
        #             output['ndcgs_'+i] = 0
        #             output['hits_'+i] = 0
        #             output['mrrs_'+i] = 0
        #         else:
        #             output['ndcgs_'+i] = y_correct_ndcg[i]/y_total_ndcg[i] # 数量
        #             output['hits_'+i] = y_correct_hit[i]/y_total_hit[i]
        #             output['mrrs_'+i] = y_correct_mrr[i]/y_total_mrr[i]
    else:
        output['auc'] = get_metric(out, 'auc')
    # for i in output.keys():
    #     output[i] = output[i].detach().cpu().numpy()
    return output

def get_metric(out, metric, rank=None):
    # if rank is not None
        # out= out[:rank] # B, rank [16,13,12,11,0]
    if metric == 'hit':
        # print(out, getHitRatio(out, 0))
        return getHitRatio(out, rank)
    elif metric == 'ndcg':
        return getNDCG(out, rank)
    elif metric == 'mrr':
        return getMRR(out, rank)
    elif metric == 'auc':
        return getAUC(out)
    else:
        raise ValueError("Invalid metric")

def load_data(filename):
    try:
        with open(filename, "rb") as f:
            x= pickle.load(f)
    except:
        x = []
    return x

def save_data(data,filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def lr_poly(base_lr, iter, max_iter, power):
    if iter > max_iter:
        iter = iter % max_iter
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, max_iter):
    lr = lr_poly(lr, i_iter, max_iter, 0.9) # power=0.9
    optimizer.param_groups[0]['lr'] = np.around(lr,5)
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def creat_feed(batch_data, device, test=False):
    """返回torch格式的数据"""
    if test:
        u, seq, pos, seqcxt, poscxt = batch_data
        u = torch.tensor(np.array(u), dtype=torch.long).to(device)
        seq = torch.tensor(np.array(seq), dtype=torch.long).to(device)
        pos = torch.tensor(np.array(pos),  dtype=torch.long).to(device)
        # seqcxt = (torch.arange(seq.shape[1]).to(device) ,torch.tensor(np.array(seqcxt)).to(device)) # 这里直接concat？
        seqcxt = (torch.arange(seq.shape[-1]).repeat(seq.shape[0],1,1).to(device) ,torch.tensor(np.array(seqcxt)).to(device)) # 这里直接concat？
        # poscxt = (torch.arange(seq.shape[1]).to(device) ,torch.tensor(np.array(poscxt)).to(device))
        poscxt = (torch.arange(seq.shape[-1]).repeat(seq.shape[0],1,1).to(device) ,torch.tensor(np.array(poscxt)).to(device))
        return (u, seq, pos, seqcxt, poscxt)
    else:
        u, seq, pos, neg, seqcxt, poscxt, negcxt = batch_data
        u = torch.tensor(np.array(u), dtype=torch.long).to(device)
        seq = torch.tensor(np.array(seq), dtype=torch.long).to(device)
        pos = torch.tensor(np.array(pos),  dtype=torch.long).to(device)
        neg = torch.tensor(np.array(neg), dtype=torch.long).to(device)
        seqcxt = (torch.arange(seq.shape[1]).to(device) ,torch.tensor(np.array(seqcxt)).to(device)) # 这里直接concat？
        poscxt = (torch.arange(seq.shape[1]).to(device) ,torch.tensor(np.array(poscxt)).to(device))
        negcxt = (torch.arange(seq.shape[1]).to(device) ,torch.tensor(np.array(negcxt)).to(device))
        return (u, seq, pos, neg, seqcxt, poscxt, negcxt)


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def get_last_visit(hidden_states, mask):
    """Gets the last visit from the sequence model.

    Args:
        hidden_states: [batch size, seq len, hidden_size]
        mask: [batch size, seq len]

    Returns:
        last_visit: [batch size, hidden_size]
    """
    if mask is None:
        return hidden_states[:, -1, :]
    else:
        mask = mask.long()
        last_visit = torch.sum(mask, 1) - 1
        last_visit = last_visit.unsqueeze(-1)
        last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
        last_visit = torch.reshape(last_visit, hidden_states.shape)
        last_hidden_states = torch.gather(hidden_states, 1, last_visit)
        last_hidden_state = last_hidden_states[:, 0, :]
        return last_hidden_state
