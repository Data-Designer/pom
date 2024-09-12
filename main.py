# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : main.py
# Time       ：13/5/2024 10:05 am
# Author     ：XXXX
# version    ：python 
# Description：
"""
import json
import numpy as np
from trainer import train
from data_pro import get_dataloader
from trainer import evalu

def run_single_config(name):
    if name == 'bst':
        from bst import BST as Model
        from bst import opt
    elif name == 'cacar':
        from cacar import CACAR as Model
        from cacar import opt
    elif name == 'caser':
        from caser import CASER as Model
        from caser import opt
    elif name == 'dien':
        from dien import DIEN as Model
        from dien import opt
    elif name == 'din':
        from din import DIN as Model
        from din import opt
    elif name == 'dupn':
        from dupn import DUPN as Model
        from dupn import opt
    elif name == 'mojito':
        from mojito import MOJOTO as Model
        from mojito import opt
    elif name == 'poprec':
        from poprec import PopRec as Model
        from poprec import opt
    elif name == 'shan':
        from shan import SHAN as Model
        from shan import opt
    elif name == 'ssl':
        from ssl import SSLPT as Model
        from ssl import opt
    elif name == 'stamp':
        from stamp import STAMP as Model
        from stamp import opt
    elif name == 'ours':
        from ours import FR as Model
        from ours import opt
    elif name == 'ours_multi':
        from ours_multi import FR as Model
        from ours_multi import opt
    elif name == 'ours_swap':
        from ours_swap import FR as Model
        from ours_swap import opt

    print("trainer test")
    if name in ['ours', 'dupn', 'dien','ours_multi']: # 感觉是加强采样出了问题啊
        sampler, test_sampler, num_batch, ItemFeatures, usernum, itemnum = get_dataloader(opt.dataset_name, opt, win=True, forw=True)
    else:
        sampler, test_sampler, num_batch, ItemFeatures, usernum, itemnum = get_dataloader(opt.dataset_name, opt, win=True, forw=False)
    # u, seq, pos, neg, seqcxt, poscxt, negcxt  = sampler.next_batch()
    # print(u[0], seq[0], pos[0], neg[0], seqcxt[0], poscxt[0], negcxt[0])
    # for i in test_sampler:
    #     print('A', i)
    #     break
    print(ItemFeatures.shape)
    UserFeatures = np.random.randn(usernum, opt.hidden_units)  # 10个商品
    GraphFeatures = np.random.randn(itemnum+1, opt.hidden_units)  # 因为是从0开始编码的,所以要加衣
    # print(GraphFeatures.shape)
    # GraphFeatures = np.loadtxt('/home/xxx/pom/data/ready/'+opt.dataset_name+'/item_graph_emb.csv', delimiter=',')
    # print(GraphFeatures.shape)
    # with open('/home/xxx/pom/data/ready/'+opt.dataset_name+ '/attribute_dict.json', 'r') as json_file:
    #     attribute_dict = json.load(json_file)
    attribute_dict = None#{}

    # print("A", ItemFeatures.shape, GraphFeatures.shape)
    print("Config", opt)
    if name in ['ours', 'ours_multi']:
        model = Model(usernum=usernum, itemnum=itemnum, opt=opt, ItemFeatures=ItemFeatures, GraphFeatures=GraphFeatures, cxt_size=6,
                      use_res=False)

    else:
        model = Model(usernum=usernum, itemnum=itemnum, opt=opt, ItemFeatures=ItemFeatures, UserFeatures=UserFeatures, cxt_size=6,
                      use_res=False)
    print("Success load! ")
    opt.gpu = '1'

    train(opt, model, (sampler, test_sampler, num_batch), attribute_dict)
    model_path = opt.model_path + opt.model_name + '-' + opt.dataset_name + '.pt'

    evalu(opt, model, test_sampler, model_path)



if __name__ == '__main__':
    run_single_config('ours') # ssl, bst, din, dupn, # shan, stamp,caser
