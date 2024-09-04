# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : graph.py
# Time       ：11/6/2024 8:27 am
# Author     ：Chuang Zhao
# version    ：python 
# Description：预处理数据，构建图
"""
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dgl
import os
import numpy


from tqdm import tqdm
import torch
import  numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init # 初始化
from torch.utils.data import Dataset, DataLoader
# from data_pro import get_ItemData # cycle import error
from tqdm import trange

class CommonArgs:
    NUM_WALKS_PER_NODE = 7
    WALK_LENGTH = 4
    root = '/home/czhaobo/HyperHealth/src/pom/data/ready/'
    root_to = '/home/czhaobo/HyperHealth/src/pom/data/ready/'

    NUM_WALKS_PER_NODE = 100  # 每个节点随机游走的次数
    WALK_LENGTH = 4  # MetaPath*it的长度,用于采样
    MIN_COUNT = 0  # 最小出现次数，我们这里不存在
    CARE_TYPE = 0  # care_type为0则不考虑节点类型
    WINDOW_SIZE = 7  # 滑动窗口的大小
    BATCH_SIZE = 50  # 每次送入skipgram的batch大小
    ITERATIONS = 1  # Epochs，常用为5，这里仅为for fluent
    INITIAL_LR = 0.025  # lr
    DIM = 32  # 这个要根据主model的设置，需要和主model相一致
    NUM_WORKER = 12

opt_graph = CommonArgs()

def graph_preprocess():
    video_graph()
    # food_graph()
    # netease_graph()
    # steam_graph()

def generate_func_file(data, path):
    """生成inter文件"""
    func = data['func'].unique()
    func = pd.DataFrame(func, columns=['func'])
    func['id'] = range(1, len(func) + 1)
    func.to_csv(path + 'id_func.txt', index=False, header=False)

    item = data['item'].unique()
    item = pd.DataFrame(item, columns=['item'])
    item['id'] = range(1, len(item) + 1)
    item.to_csv(path + 'id_item.txt', index=False, header=False)

    # 交互
    func_item = data[['func', 'item']]
    func_item.to_csv(path + 'func_item.txt', index=False, header=False)
    print("Generate func file Done!")

# def video_graph():
#     path = os.path.join(opt_graph.root, 'video')
#     feat = get_ItemData(path, 'video')
#     # 生成id_item和交互文件
#     generate_func_file(feat, path)
#
#     generate_metapath(path)
#     print("video graph generated!")
#     # 开始训练
#     print("Start trainning graph!")
#     model = Metapath2VecTrainer(path)
#     model.train()
#     print("Embedding have been trained !")
#     convert_name2id(path)
#     print("Convert Done !")

def video_graph(path, df, column):
    # 获取interaction pairs
    interactions = []

    for i, row in df.iterrows():
        row_lis = row[column].tolist()[0]
        for attr in row_lis:
            interactions.append([i+1, attr.strip()])
    df = pd.DataFrame(interactions, columns=['item', 'func'])
    unique_values = df['func'].unique()
    encode_dict = {value: i for i, value in enumerate(unique_values,start=1)} # 从0开始，name:index
    # 应用编码字典
    df['func_encoded'] = df['func'].map(encode_dict)
    df[['func_encoded', 'item']].to_csv(os.path.join(path, 'func_item.txt'), index=False, header=False)
    # 生成id_func.txt
    id_func = pd.DataFrame(list(encode_dict.items()), columns=['func', 'id'])
    id_func[['id', 'func']].to_csv(os.path.join(path, 'id_func.txt'), index=False, header=False)


    generate_metapath(path)

    print("video graph generated!")
    # 开始训练
    print("Start trainning graph!")
    model = Metapath2VecTrainer(path)
    model.train()

    print("Embedding have been trained !")
    convert_name2id(path)
    print("Convert Done !")





def food_graph():
    path = os.path.join(opt_graph.root, 'food')
    feat = get_ItemData(path, 'food')
    # 生成id_item和交互文件
    generate_func_file(feat, path)

    generate_metapath(path)
    print("food graph generated!")
    # 开始训练
    print("Start trainning graph!")
    model = Metapath2VecTrainer(path)
    model.train()
    print("Embedding have been trained !")
    convert_name2id(path)
    print("Convert Done !")

def netease_graph(path, df, column):
    # 获取interaction pairs
    print('attribute matrix shape', df.shape)

    interactions = []
    for i, row in df.iterrows():
        # print(row[column])
        # print(row[column][0])
        for attr in row[column][0]:
            interactions.append([i+1, attr])
    df = pd.DataFrame(interactions, columns=['item', 'func'])
    unique_values = df['func'].unique()
    encode_dict = {value: i for i, value in enumerate(unique_values,start=1)} # 从0开始，name:index
    # 应用编码字典
    df['func_encoded'] = df['func'].map(encode_dict)
    df[['func_encoded', 'item']].to_csv(os.path.join(path, 'func_item.txt'), index=False, header=False)
    # 生成id_func.txt
    id_func = pd.DataFrame(list(encode_dict.items()), columns=['func', 'id'])
    id_func[['id', 'func']].to_csv(os.path.join(path, 'id_func.txt'), index=False, header=False)
    # path = os.path.join(opt_graph.root, 'netease')
    # feat = get_ItemData(path, 'netease')
    # 生成id_item和交互文件
    # generate_func_file(feat, path)


    generate_metapath(path)
    print("netease graph generated!")
    # 开始训练
    print("Start trainning graph!")
    model = Metapath2VecTrainer(path)
    model.train()
    print("Embedding have been trained !")
    convert_name2id(path)
    print("Convert Done !")


def steam_graph():
    path = os.path.join(opt_graph.root, 'steam')
    feat = get_ItemData(path, 'steam')
    # 生成id_item和交互文件
    generate_func_file(feat, path)


    generate_metapath(path)
    print("steam graph generated!")
    # 开始训练
    print("Start trainning graph!")
    model = Metapath2VecTrainer(path)
    model.train()
    print("Embedding have been trained !")
    convert_name2id(path)
    print("Convert Done !")

def construct_graph(path):
    """生成图和ID对照表, 这里的func其实是item, item是ingredient"""
    func_ids = []
    func_names = [] # reindex后的序号和这个是对应的。
    item_ids = []
    item_names = []

    # # 这两个文件需要在预处理阶段准备好
    file_func = open(os.path.join(path, 'id_func.txt'),encoding='utf-8') # str[ids], str[name]
    file_item = open(os.path.join(path, 'id_item.txt'),encoding='utf-8')

    # 读取文件
    while True:
        line_tmp = file_func.readline()
        if not line_tmp:
            break
        line_tmp = line_tmp.strip().split(',') # lis
        func_id = int(line_tmp[0])
        func_ids.append(func_id)
        func_names.append(line_tmp[1])
    while True:
        line_tmp = file_item.readline()
        if not line_tmp:
            break
        line_tmp = line_tmp.strip().split(',')
        item_id = int(line_tmp[0])
        item_ids.append(item_id)
        item_names.append(line_tmp[1])
    file_item.close()
    file_func.close()
    # 防止有删减id的情况发生
    func_ids_reindex = {id: index for index, id in enumerate(func_ids)} # raw_id -> new_index, start from 0
    item_ids_reindex = {id: index for index, id in enumerate(item_ids)}
    # print('item_ids,XXXXXXXXXXXXXXXXXXXXXXXXXXXX', item_ids)
    # print('func_ids', func_ids_reindex)

    func_item_src = []
    func_item_dst = []
    file_action = open(os.path.join(path,'func_item.txt'), "r")
    for line_tmp in file_action:
        line_tmp = line_tmp.strip().split(',')
        func_raw_id = int(line_tmp[0].strip())
        item_raw_id = int(line_tmp[1].strip())
        func_item_src.append(func_ids_reindex[func_raw_id])
        func_item_dst.append(item_ids_reindex[item_raw_id])
    file_action.close()
    # construct graph
    hg = dgl.heterograph({
        ('func','fi','item'): (func_item_src, func_item_dst), # 构建图
        ('item','if','func'): (func_item_dst, func_item_src)
    })
    return hg, func_names, item_names # graph, 这里func不能用中文名, dict


def generate_metapath(path):
    """生成图对应的metapath"""
    output_path = open(os.path.join(path, 'output_path.txt'), "w")

    hg, func_names, item_names = construct_graph(path)
    for func_idx in trange(hg.number_of_nodes('func')):
        traces, _ = dgl.sampling.random_walk(
            hg, [func_idx] * opt_graph.NUM_WALKS_PER_NODE, metapath=['fi','if'] * opt_graph.WALK_LENGTH
        )
        # 这里可以对metapath做一些操作, 将每一条path居然恢复为了对应的name
        for tr in traces:
            outline = '+'.join([(func_names if i%2==0 else item_names)[tr[i]] for i in range(0,len(tr))])
            print(outline, file=output_path)
    output_path.close() # file 使用完必须关闭




def data_process(x, y, feature_index):
    """
    :param: 返回TensorData所需要的格式
    :return:
    """
    if isinstance(x, dict):
        x = [x[feature] for feature in feature_index]
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)  # 维度为1 的扩展
    x = torch.from_numpy(np.concatenate(x, axis=-1))  # 一个人就是一条信息
    y = torch.from_numpy(y).long()
    return x, y



class CustomDataset(object):
    """
    Custom dataset generated by sampler.py (e.g. NetDBIS)
    """
    def __init__(self, path):
        self.fn = path
        print("Custom data path!", path)

class DataReader:
    """根据元路径-》生成负采样表-》【正，负】"""
    NEGATIVE_TABLE_SIZE = 1e8
    def __init__(self, dataset, min_count, care_type):
        self.negatives = [] # negative_table
        self.discards = [] # 用于丢弃
        self.negpos = 0 # 用于定位Negative table中的负样本位置
        self.care_type = care_type
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.inputFileName = dataset.fn
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="utf-8"):
            line = line.split('+') # netease 放宽了分隔符()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word = word.strip().strip('"')
                        word_frequency[word] = word_frequency.get(word, 0) + 1 # 单词计数

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count: # 我们不存在最小出现问题，只需要后期过滤即可
                continue
            self.word2id[w] = wid # 不管是recipe还是item统一编码
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        self.word_count = len(self.word2id)
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        # get a frequency table for sub-sampling. Note that the frequency is adjusted by
        # sub-sampling tricks.
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count # norm freq
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        # get a table for negative sampling, if word with index 2 appears twice, then 2 will be listed
        # in the table twice.
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def getNegatives(self, target, size):  # TODO check equality with target
        if self.care_type == 0:
            response = self.negatives[self.negpos:self.negpos + size]
            self.negpos = (self.negpos + size) % len(self.negatives)
            if len(response) != size:
                return np.concatenate((response, self.negatives[0:self.negpos])) # 负样本超过表长
        return response




class Metapath2vecDataset(Dataset):
    """MetaPath Dataset构造"""
    def __init__(self, data, window_size):
        # read in data, window_size and input filename
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, 'r', encoding='unicode_escape') # 也是要重新编码的，所以上面无所谓

    def __len__(self):
        # return the number of walks
        return self.data.sentences_count

    def __getitem__(self, idx):
        # return (center, context, 5 negatives[list])
        while True:
            line = self.input_file.readline()
            # print("line", line)
            if not line:
                self.input_file.seek(0,0) # 不足batch则用头补上
                line = self.input_file.readline()

            line = line.strip()


            if len(line) >1:
                words = line.split('+') #  [recipe_name1  rabbit  recipe_name1  rabbit  recipe_name1]
                # print('words', words)
                # words[-1] = words[-1].strip() # 可能会有\n
                words = [i.strip().strip('"') for i in words] # [recipe_name1, recipe_name1, recipe_name1

                if len(words) >1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]
                    pair_catch = []
                    for i , u in enumerate(word_ids): # center
                        for j, v in enumerate(word_ids[max(i-self.window_size,0):i+self.window_size]):
                            # neighbors
                            assert u < self.data.word_count
                            assert v < self.data.word_count
                            if i==j:
                                continue
                            pair_catch.append((u,v,self.data.getNegatives(v,2))) # 这里的负采样数量可以自定义
                    return pair_catch

    @staticmethod
    def collate(batches):
        """用于处理一个batch的数据"""
        all_u = [u for batch in batches for u,_,_ in batch if len(batch)>0]
        all_v = [v for batch in batches for _,v,_ in batch if len(batch)>0]
        all_neg_v = [neg_v for batch in batches for _,_,neg_v in batch if len(batch)>0]
        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)


class SkipGramModel(nn.Module): # TODO
    """图使用的model"""
    def __init__(self,emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size # vocab-size
        self.emb_dimension = emb_dimension # emb_dim
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True) # center word,这里两种emb不共享
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True) # neighbor word
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u) # B, H [注意这里的H需要和主model的item_emb+cate_emb大小相一致]
        emb_v = self.v_embeddings(pos_v) # B, Window, H
        emb_neg_v = self.v_embeddings(neg_v) # B, Neg, H

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score) # Batch sum

    def save_embedding(self, id2word, file_name):
        """将embedding存入文件"""
        embedding = self.u_embeddings.weight.cpu().data.numpy() # 必须转到cpu才能写入文件
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s+%s\n' %(w, e)) # 这里存储的好像是name,因为recipe和item会统一编码，在这里使用id会乱





class Metapath2VecTrainer:
    """生成graph embedding"""
    def __init__(self, path):
        dataset = CustomDataset(path + '/output_path.txt') # meta-path
        self.output_file_name = path + '/output_emb.txt' # 最终得到的emb
        self.data = DataReader(dataset, opt_graph.MIN_COUNT,opt_graph.CARE_TYPE) # 构造负采样表
        dataset = Metapath2vecDataset(self.data,opt_graph.WINDOW_SIZE)
        self.dataloader = DataLoader(dataset,batch_size=opt_graph.BATCH_SIZE,shuffle=True, num_workers= opt_graph.NUM_WORKER, collate_fn=dataset.collate)

        self.emb_size = len(self.data.word2id) # vocab
        self.emb_dimension = opt_graph.DIM
        self.batch_size = opt_graph.BATCH_SIZE
        self.iterations = opt_graph.ITERATIONS
        self.initial_lr = opt_graph.INITIAL_LR
        self.skip_gram_model = SkipGramModel(self.emb_size,self.emb_dimension)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):
        optmizer = torch.optim.SparseAdam(list(self.skip_gram_model.parameters()), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optmizer,len(self.dataloader))

        for iteration in range(self.iterations):
            print("\n\n\nIteration："+ str(iteration + 1))
            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                # print("sample batched", sample_batched)

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    optmizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u,pos_v,neg_v)
                    loss.backward()
                    optmizer.step()
                    scheduler.step()


                    running_loss = running_loss * 0.9 + loss.item() * 0.1 # 平均也行,这也是一种看收敛的方式
                    if i >0 and i%500 ==0:
                        print(" Loss: "+ str(running_loss))

        self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


def convert_name2id(path):
    # 获取对照id表
    data_item = pd.read_table(path + 'id_item.txt', sep=',', header=None)
    data_func = pd.read_table(path+ 'id_func.txt', sep=',', header=None)
    data_item.columns =["id",'item_name'] # 原来的id，map id (用train)
    data_func.columns =["id",'func_name'] # 原来的id，map id (用train)
    # 转换output_emb表,转换recipe_name为id
    data = pd.read_table(path+'output_emb.txt', sep='\t', header=0)
    print(data.head())
    res = {} # 存储读取的embed {name:emb}
    for line in range(data.shape[0]): #
        line_data = data.iloc[line].values[0].strip().split("+")
        try:
            line_data = [line_data[0], line_data[1].split(' ')] # 有的名字里带空格
            # print(line_data)
            func_name, emb = line_data[0], ','.join(line_data[1])
            res[func_name] = emb
        except:
            print("line data", line_data)
            break
    # print(res.keys())
    # print('XXXXXXX', res['B000F9JFXE']) # 这里不对啊，这里明明是func啊；
    print("len item, fun,res,",  data_item.shape, data_func.shape, len(res))
    # for key in data_func['func_name']:
    #     try:
    #         list(eval(res[key]))
    #     except:
    #         print('key', res[key])
    data_func['emb'] = data_func['func_name'].apply(lambda x: list(eval(res[x])) if x in res else [0] * 32)
    # data_func['emb'] = data_func['func_name'].apply(lambda x: list(eval(res[x]))) # 注意在后期处理的时候item需要加1，并且加一行padding的item
    # data_func = data_func.sort_values(by="id", ascending=True)
    # data_func = data_func.sort_values(by="func_name", ascending=True) # 因为这个最有用,为啥要sort
    data_func['len'] = data_func['emb'].apply(lambda x: len(x))

    emb = np.array(list(data_func['emb']),dtype='float')
    emb = np.insert(emb, 0, values=np.random.rand(1, opt_graph.DIM), axis=0) # padding的emb
    np.savetxt(path + "func_graph_emb.csv", emb, delimiter=',')

    print('func_graph_emb.csv saved!, shape:', emb.shape)

    # 下面是item的emb
    data_item['emb'] = data_item['item_name'].apply(lambda x: list(eval(res[x])) if x in res else [0] * 32)
    # data_func = data_func.sort_values(by="func_name", ascending=True) # 因为这个最有用,为啥要sort
    data_item['len'] = data_item['emb'].apply(lambda x: len(x))

    emb = np.array(list(data_item['emb']),dtype='float')
    emb = np.insert(emb, 0, values=np.random.rand(1, opt_graph.DIM), axis=0) # padding的emb
    np.savetxt(path + "item_graph_emb.csv", emb, delimiter=',')

    print('item_graph_emb.csv saved!, shape:', emb.shape)



if __name__ == '__main__':
    # 预处理
    convert_name2id('/home/czhaobo/pom/data/ready/video/')


