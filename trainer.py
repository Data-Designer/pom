# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : trainer.py
# Time       ：6/5/2024 8:22 am
# Author     ：Chuang Zhao
# version    ：python 
# Description：
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import datetime
# import torchmetrics
import pandas as pd
from utils import cal_metrics, load_data,adjust_learning_rate, pearson_corr, cosine, euclidean, spearman_rank_correlation
from data_pro import get_dataloader
from tqdm import tqdm
from utils import creat_feed
from collections import defaultdict



########### train define
def calculate_loss(model, output, pos, criterion, monitor_metrics=False, reg=True, test=False):
    loss, metric = 0, 0
    if test:
        label = torch.zeros((output.shape[0], output.shape[-1])).to(output.device) # 一个正样本，99个阜阳本
        label[:, 0] = 1
        pos_logits = output
        loss = -torch.log(label * torch.sigmoid(pos_logits) + 1e-24).sum() / output.shape[-1]
        return loss
    else:
        pos_logits, neg_logits = output[:,:,0], output[:,:,1]
        # pos_labels, neg_labels = torch.ones(pos.shape, device=pos.device), torch.zeros(pos.shape,device=pos.device)
        # indices = torch.nonzero(pos != 0)
        # loss = criterion(pos_logits[indices], pos_labels[indices])
        # loss += criterion(neg_logits[indices], neg_labels[indices])
        istarget = (pos != 0).float()# .view(-1)
        # print(indices, istarget)
        loss = (
                       -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
                       torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
               ).sum() / istarget.sum()
        if monitor_metrics:
            sign_diff = (torch.sign(pos_logits - neg_logits) + 1) / 2
            # 通过乘以 istarget 掩码来筛选目标数据点
            targeted_diff = sign_diff * istarget
            # 计算 AUC，即目标数据点的平均值
            metric = torch.sum(targeted_diff) / torch.sum(istarget)
        if reg:
            # 添加正则化损失
            # reg_losses = [param.norm(2) for param in model.parameters()]  # 假设这是模型中的所有参数的 L2 正则化项
            # loss += sum(reg_losses)
            for param in model.parameters(): loss += 0.0001 * torch.norm(param)
    return loss, metric


def train_step(model, batch_data, monitor_metrics, criterion, optimizer, reg=False):
    model.train()
    u, seq, pos, neg, seqcxt, poscxt, negcxt = batch_data
    optimizer.zero_grad()
    output, aux = model(u, seq, pos, neg, seqcxt, poscxt, negcxt)
    loss, metric = calculate_loss(model, output, pos, criterion, monitor_metrics=monitor_metrics, reg=reg) # 这种是序列损失；
    loss = loss + aux
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 4
    )
    optimizer.step()
    return loss.item(), metric

def evaluate(model, tes_loader, criterion=None, device=None, attribute_dict=None):
    losses_lis = []
    rational_embs = []
    emotional_embs = []
    item_rat_embs = []
    metrics_lis = defaultdict(list)
    model.eval()
    with torch.no_grad():
        print("test loader num: ",len(tes_loader))
        for batch_data in tes_loader:#range(len(tes_loader)):
            # batch_data = tes_loader[i]
            # print("A", batch_data[0].shape, batch_data[1].shape, batch_data[2].shape, batch_data[3].shape, batch_data[4].shape, batch_data[5].shape)
            # print(batch_data)
            labels, seq, item_idx, seq_cxt, testitemscxt = creat_feed(batch_data, device, test=True)
            # print(labels.shape)
            # print(seq.shape)
            # print(item_idx.shape)
            # # print(seq_cxt)
            # print(seq_cxt[0].shape,seq_cxt[1].shape)
            # print(testitemscxt[0].shape, testitemscxt[1].shape)
            '''
            torch.Size([2, 50])                                                   
            torch.Size([2, 1, 50])
            torch.Size([2, 101])
            torch.Size([2, 1, 50]) torch.Size([2, 1, 50, 6])
            torch.Size([2, 1, 50]) torch.Size([2, 101, 6])
            '''

            output = model.predict(labels, seq, item_idx, seq_cxt, testitemscxt) # B, 100

            # rational_embs.append(output['rational'].cpu().numpy())
            # emotional_embs.append(output['emotional'].cpu().numpy())
            # item_rat_embs.append(output['item_rational'].cpu().numpy())
            output = output['logit']

            loss = calculate_loss(model, output, item_idx, criterion, reg=False, test=True) # 100,1
            losses_lis.append(loss)
            output = output.squeeze() # B,100
            rank_lis = (-output).argsort().argsort() # [0.7, 0.1,0,1]->[0,1,2]

            # teshu
            sorted_indices = torch.argsort(output, dim=1, descending=True)
            rank_items = torch.gather(item_idx, dim=1, index=sorted_indices) # 重排好的物品

            metrics = cal_metrics(output, rank_lis, test=True, y_group=attribute_dict,rank_items=rank_items) # {'hit-5':0.1}
            for key in metrics.keys():
                metrics_lis[key].append(metrics[key]) # [0.1,0.2,0.1...]
        loss = sum(losses_lis) / len(losses_lis)
        for key in metrics.keys():
            metrics[key] = sum(metrics_lis[key]) / len(metrics_lis[key])
        metrics['loss'] = loss.item()

        # rational_embs = np.concatenate(rational_embs, axis=0) # S,D
        # emotional_embs = np.concatenate(emotional_embs, axis=0)
        # item_rat_embs = np.concatenate(item_rat_embs, axis=0)
        #
        # print('Caculate Pearson Correlation', pearson_corr(rational_embs, item_rat_embs))
        # print('Caculate Cosine Similarity', cosine(rational_embs, item_rat_embs))
        # print('Caculate Spearman Distance', spearman_rank_correlation(rational_embs, item_rat_embs))
        #
        # print('Caculate Pearson Correlation', pearson_corr(emotional_embs, item_rat_embs))
        # print('Caculate Cosine Similarity', cosine(emotional_embs, item_rat_embs))
        # print('Caculate Spearman Distance', spearman_rank_correlation(emotional_embs, item_rat_embs))


    return metrics

def train(opt, model, data, attribute_dict=None):
    # 参数解析
    device = 'cpu'
    use_cuda = opt.gpu_used
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:' + opt.gpu

    # 数据读取、模型定义
    # 数据处理,设立日志记录
    sampler, test_sampler, num_batch = data
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    # criterion = torch.nn.CrossEntropyLoss()
    bce_criterion = torch.nn.BCEWithLogitsLoss()

    monitor_metric = 'auc'
    dfhistory = pd.DataFrame(columns=['epoch', 'loss', 'auc','hit_1', 'ndcg_1', 'mrr_1', 'hit_5', 'ndcg_5', 'mrr_5', 'hit_10', 'ndcg_10', 'mrr_10',
                                      'hit_15', 'ndcg_15', 'mrr_15', 'hit_20', 'ndcg_20', 'mrr_20'])
    print("Start Training !")
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 8 + "%s" % now_time)

    # 开始训练
    best_auc = 0.0
    for epoch in range(1, opt.num_epochs + 1):
        loss_sum = 0.0
        metric_sum = 0.0
        cur_step = 0
        # for step, (features, labels) in enumerate(sampler, 1):  # x=[uid,item_id,cate_id,his_item,his_cate,]
        for step in tqdm(range(int(num_batch)), total=int(num_batch), ncols=70, leave=False, unit='b'):
            cur_step += 1
            adjust_learning_rate(optimizer, cur_step, opt.lr, num_batch)  # 真的离谱
            batch_data = sampler.next_batch()
            batch_data = creat_feed(batch_data, device)
            loss, metric = train_step(model, batch_data, monitor_metric, bce_criterion, optimizer, reg=opt.reg)
            loss_sum += loss
            metric_sum += metric

            if step!=0 and step % opt.log_freq == 0:
                print(("[step = %d] loss: %.3f, " + monitor_metric + ": %.3f, ") %
                      (step, loss_sum / step, metric_sum / step))

        # 下面是valid
        info = evaluate(model, test_sampler, bce_criterion, device, attribute_dict)
        dfhistory.loc[epoch - 1] = (epoch, info['loss'], info['auc'],info['hit_1'], info['ndcg_1'], info['mrr_1'], info['hit_5'], info['ndcg_5'], info['mrr_5'],
        info['hit_10'], info['ndcg_10'], info['mrr_10'], info['hit_15'], info['ndcg_15'], info['mrr_15'],
                                    info['hit_20'], info['ndcg_20'], info['mrr_20'])
        if info['hit_5'] > best_auc:
            best_auc = info['hit_5'] # 用hit
            torch.save(model.state_dict(), opt.model_path + opt.model_name + '-' + opt.dataset_name + '.pt')

        # 打印信息
        print(("\nEPOCH = %d, loss = %.4f, " + monitor_metric + " = %.4f," + \
               "\nval_loss = %.4f," + "val_" + "AUC " +
               " = %.4f, " + "Hit@10 " + " = %.4f, " + "NDCG@10 " +
               " = %.4f, " + "MRR@10 " + " = %.4f, ") % (epoch , loss_sum / step, metric_sum / step,
                info['loss'], info['auc'], info['hit_10'], info['ndcg_10'], info['mrr_10']))

        print("Rank 1:",info['hit_1'], info['ndcg_1'], info['mrr_1'])
        print("Rank 5:",info['hit_5'], info['ndcg_5'], info['mrr_5'])
        print("Rank 15:",info['hit_15'], info['ndcg_15'], info['mrr_15'])
        print("Rank 20:",info['hit_20'], info['ndcg_20'], info['mrr_20'])

        # if attribute_dict is not None:
        #     for i in attribute_dict.keys():
        #         print(i, info['hits_'+i], info['ndcgs_'+i], info['mrrs_'+i])

        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % now_time)

    print("Finished Training...")
    dfhistory.to_csv(opt.log_path + opt.model_name + '-' + opt.dataset_name + '.csv')
    print("LOG FILE HAVE SAVED!")



def evalu(opt, model, data, model_path):
    # 参数解析
    device = 'cpu'
    use_cuda = opt.gpu_used
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:' + opt.gpu

    # 数据读取、模型定义
    state_dict = torch.load(model_path, map_location=device)
    model = model.load_state_dict(state_dict, strict=False)

    # 数据处理,设立日志记录
    test_sampler = data
    info = evaluate(model, test_sampler, None, device)
    # 打印信息
    print(("\nval_loss = %.4f," + "val_" + "AUC " +
               " = %.4f, " + "Hit@10 " + " = %.4f, " + "NDCG@10 " +
               " = %.4f, " + "MRR@10 " + " = %.4f, ") % (
                info['loss'], info['auc'], info['hit_10'], info['ndcg_10'], info['mrr_10']))
    print("Finished Inference...")

if __name__ == '__main__':
    pass
    # name = 'bst'
    # if name == 'bst':
    #     from bst import BST as Model
    #     from bst import opt
    # elif name == 'cacar':
    #     from cacar import CACAR as Model
    #     from cacar import opt
    # elif name == 'caser':
    #     from caser import Caser as Model
    #     from caser import opt
    # elif name == 'dien':
    #     from dien import DIEN as Model
    #     from dien import opt
    # elif name == 'din':
    #     from din import DIN as Model
    #     from din import opt
    # elif name == 'dupn':
    #     from dupn import DUPN as Model
    #     from dupn import opt
    # elif name == 'mojito':
    #     from mojito import MOJOTO as Model
    #     from mojito import opt
    # elif name =='poprec':
    #     from poprec import PopRec as Model
    #     from poprec import opt
    # elif name == 'shan':
    #     from shan import SHAN as Model
    #     from shan import opt
    # elif name == 'ssl':
    #     from ssl import SSLPT as Model
    #     from ssl import opt
    # elif name == 'stamp':
    #     from stamp import STAMP as Model
    #     from stamp import opt
    # elif name == 'ours':
    #     from ours import FR as Model
    #     from ours import opt
    #
    # print("trainer test")
    # ItemFeatures = np.random.randn(30000, 506)  # 10个商品
    # GraphFeatures = np.random.randn(50000, 90)  # 10个商品
    # UserFeatures = np.random.randn(50000, 90)  # 10个商品
    # model = Model(usernum=50000, itemnum=30000, opt=opt, ItemFeatures=ItemFeatures, UserFeatures=UserFeatures, cxt_size=6,
    #               use_res=False)
    # print("Success load! ")
    # train(opt, model)
    #
