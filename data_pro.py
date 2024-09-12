# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : data_pro.py
# Time       ：29/4/2024 7:53 pm
# Author     ：XXXX
# version    ：python 
# Description：Steam的格式有点东西哈
# 这里的Stem的games_csv包含了一些游戏的原信息
# recommendations表明用户是否推荐该游戏
还有计数；
"""
from graph import video_graph, netease_graph
import random
import copy
import json
import array
import itertools

import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process, Queue
from utils import parse, load_data, save_data
import ast
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
import torch


def get_bert_embeddings(texts, model_name='bert-base-chinese'):
    save_directory = "/home/xxx/pom/data/ready/netease/"

    tokenizer = BertTokenizer.from_pretrained(save_directory)
    model = BertModel.from_pretrained(save_directory)

    embeddings = []
    for text in texts:
        # print(text)
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # 取[CLS] token的embedding作为文本的表示
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)

    return np.array(embeddings)


def assign_clusters_to_items(merge_df, num_clusters=6):
    # 处理每个item的属性列表

    all_attributes = merge_df['attribute'].explode().unique()
    all_attributes = [attr for attr in all_attributes if isinstance(attr, str)]

    # print("all_attributes", all_attributes)
    # 获取属性的BERT嵌入
    embeddings = get_bert_embeddings(all_attributes)

    # 对嵌入进行KMeans聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # 创建属性到聚类ID的映射
    attr_to_cluster = dict(zip(all_attributes, cluster_labels))

    # 为每个item_id分配聚类ID
    def get_item_cluster(attrs):
        clusters = [attr_to_cluster.get(attr) for attr in attrs if attr in attr_to_cluster]
        return max(set(clusters), key=clusters.count) if clusters else None

    merge_df['cluster_id'] = merge_df['attribute'].apply(get_item_cluster)

    # 创建聚类ID到item_id的映射
    cluster_dict = merge_df.groupby('cluster_id')['new_ID'].apply(list).to_dict()
    all_id = set(list(range(316))) # max_item
    existing_items = set()
    for itemlist in cluster_dict.values():
        existing_items.update(itemlist)
    cluster_dict['6.0']  = list(all_id - existing_items)
    print(cluster_dict)
    return cluster_dict


# # 示例使用
# data = {
#     'item_id': [1, 2, 3],
#     'attribute': [['A', 'B'], ['B', 'C'], ['A', 'C']]
# }
#
# merge_df = pd.DataFrame(data)
# cluster_dict = assign_clusters_to_items(merge_df, num_clusters=6)
# print(cluster_dict)


# from graph import video_graph, netease_graph
class CommonArgs:
    min_num = 5
    tes_batch = 10 # 这样免得出现没有
    root = '/home/xxx/pom/data/raw/'
    root_to = '/home/xxx/pom/data/ready/'
    negnum = 99

opt_com = CommonArgs()







##### 数据读取
def data_preprocess(dataset_name):
    if dataset_name == 'video': # 所有amazon dataset通用
        video_preprocess()
        video_cxt_pro()
    elif dataset_name == 'steam':
        steam_preprocess()
        steam_cxt_pro()
    elif dataset_name == 'netease':
        netease_preprocess()
        netease_cxt_pro()
    elif dataset_name == 'food':
        food_preprocess()
        food_cxt_pro()
    else:
        print("Dataset not Done!")
    print("preprocess success!")

def random_list(size, start=1, end=300):
    return [random.randint(start, end) for _ in range(size)]

# def reorder_graph_emb(root, dataset, map, save_dir):
#     """重新排序图嵌入"""
#     graph_emb = np.load(dataset)
#     graph_map = pd.read_csv(root + dataset + '/id_item.txt').to_dict()
#     new_emb = np.zeros((len(map), graph_emb.shape[1]))
#     for key, value in map.items(): # old, new
#         graph_emb = graph_map[key]
#         new_emb[value] = graph_emb[key]
#     np.save(save_dir, new_emb)
#     print("graph reorder for {}  done!".format(dataset))

def netease_preprocess():
    print("===============Processing Netease Dataset==================")


    def remove_duplicate(row):
        """his去重,consist为连续去重，all为完全去重"""
        row = [c for i, c in enumerate(row) if i == 0 or row[i - 1][0] != c[0]]
        return row
    def one_to_more(df):
        # 用列表收集处理后的DataFrame
        dfs = []
        # 遍历DataFrame的每一行
        for index, row in df.iterrows():
            # 展开列表并重新构建DataFrame
            df_temp = pd.DataFrame({
                'u': [row['role_id']] * len(row['his_item']),
                # 'his_price': row['his_price'],
                'i': row['his_item'],
                # 'his_count': row['his_count'],
                'date': row['his_ts'],
                # 'length': [row['length']] * len(row['his_item']),
            })
            # 将处理后的DataFrame添加到列表中
            dfs.append(df_temp)
        # 将列表中的DataFrame连接成一个新的DataFrame
        result_df = pd.concat(dfs, ignore_index=True)
        return result_df
    dataset_name = 'netease'
    root = opt_com.root + 'netease/'
    root_to = opt_com.root_to + 'netease/'
    line = 0
    data_1 = pd.read_csv(root + '1.csv') # 这个数据似乎是过滤好的，好像不对。
    data_2 = pd.read_csv(root + '2.csv')
    data_3 = pd.read_csv(root + '3.csv')
    data_7 = pd.read_csv(root + '4.csv')
    data_4 = pd.read_csv(root + '5.csv')
    data_5 = pd.read_csv(root + '6.csv')
    data_6 = pd.read_csv(root + '7.csv')
    df = pd.concat([data_1,data_2,data_3,data_4,data_5,data_6,data_7],ignore_index=True)
    df['his_price'] = df['his_price'].apply(ast.literal_eval)
    df['his_item'] = df['his_item'].apply(ast.literal_eval)
    df['his_count'] = df['his_count'].apply(ast.literal_eval)
    # df['his_ts'] = df['his_ts'].apply(ast.literal_eval)
    # print('XXX', df.shape, len(df['role_id'].unique()))

    #过滤非法数据
    # df['max_id'] = df['his_item'].apply(lambda x: max(x))
    # df = df[df['max_id']<=471] # 这个是为啥；
    # print('XXX', df.shape, len(df['role_id'].unique()))
    df['zip'] = df.apply(lambda x: list(zip(x['his_item'], eval(x['his_ts']))), axis=1)
    df['zip'] = df['zip'].apply(lambda x: remove_duplicate(x))
    df['his_item'] = df['zip'].apply(lambda x: list(map(lambda y:y[0],x)))
    df['his_ts'] = df['zip'].apply(lambda x: list(map(lambda y:y[1],x)))

    # df['his_item'] = df['his_item'].apply(lambda x: x[:10])
    # df['his_ts'] = df['his_ts'].apply(lambda x: x[:10])
    # df['length'] = df['his_ts'].apply(lambda x: len(x))
    # df = df[df['length']<30] # 搞一堆冷启动用户可能会好点，这里metric搞的有点问题
    df = df[['role_id', 'his_item', 'his_ts']]

    # 这里需要插入一个将其变成多行的操作
    df = one_to_more(df)

    df['date'] = pd.to_datetime(df['date'])
    df['ts'] = df['date'].apply(lambda x: x.timestamp())

    countU = df.groupby('u')['i'].count().to_dict()
    countP = df.groupby('i')['u'].count().to_dict()

    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    for index, l in df.iterrows():
        line += 1
        asin = l['i']
        rev = l['u']
        time = l['ts']
        if countU[rev] < opt_com.min_num or countP[asin] < opt_com.min_num:  # 数量, 增加netease item的个数
            continue
        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []  # 转换user_id
        if asin in itemmap: # 转换item_id
            itemid = itemmap[asin]
        else:
            itemnum += 1 # 从1开始标
            itemid = itemnum
            itemmap[asin] = itemid
        User[userid].append([time, itemid])  # 每个user id存储的是[item, timestamp]存储的是时间戳{user:[[time, id],[time. id]]}



    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    print(usernum, itemnum) # 315

    # import torch
    # torch.save({'item_map': itemmap}, root_to + 'item_map.pth')


    f = open(root_to + dataset_name + '_cxt.txt', 'w') # 带时间的
    for user in User.keys():
        for index, i in enumerate(User[user]):
            f.write('%d %d %s\n' % (user, i[1], i[0]))

    f.close()

    f = open(root_to + dataset_name + '.txt', 'w') # 存储user-id pair
    for user in User.keys():
        for index, i in enumerate(User[user]):
            f.write('%d %d\n' % (user, i[1]))
            #

    f.close()

    # read feature
    df = pd.read_json(root + 'item_info.json')
    df = df[['item_id', 'price', 'viplimit', 'buytype','firstclass', 'attribute', 'limittype']]
    # 对category列进行get dummy处理
    dummies = pd.get_dummies(df['buytype'], prefix='buytype')
    result_df = pd.concat([df, dummies], axis=1)
    result_df = result_df.drop(['buytype'], axis=1)
    dummies = pd.get_dummies(result_df['viplimit'], prefix='viplimit')
    result_df = pd.concat([result_df, dummies], axis=1)
    result_df = result_df.drop(['viplimit'], axis=1)
    dummies = pd.get_dummies(result_df['limittype'], prefix='limittype')
    result_df = pd.concat([result_df, dummies], axis=1)
    result_df = result_df.drop(['limittype'], axis=1)

    '''
    all_strings = [item for sublist in result_df['attribute'] for item in sublist]
    unique_attribute = list(set(all_strings))
    print('Unique attribute', unique_attribute)
    result_dict = {attribute: [] for attribute in unique_attribute}
    for index, row in result_df.iterrows():
        for attribute in row['attributes']:
            result_dict[attribute].append(itemmap(row['id']))
    print("Map dict success!")
    '''

    result_df['attributes'] = result_df['attribute'].apply(lambda x: len(x)) # 这里是一个list


    # 基于字典创建一个新的DataFrame
    id_df = pd.DataFrame(list(itemmap.items()), columns=['item_id', 'new_ID'])
    merged_df = id_df.merge(result_df, on='item_id', how='left', suffixes=('_dict', ''))
    merged_df[['attribute']] = merged_df[['attribute']].fillna('No')
    merged_df['attribute'] = merged_df['attribute'].apply(lambda x: ['NO'] if x == 'NO' else x)
    merged_df = merged_df.sort_values(by='new_ID', ascending=True)

    # graph_prepro
    print("Now train graph data")
    id_item = pd.DataFrame(list(itemmap.items()), columns=['item', 'id'])
    id_item['id'] = id_item['id'] # 从1开始编码
    id_item[['id', 'item']].to_csv(root_to + 'id_item.txt', index=False, header=False)
    netease_graph(root_to, merged_df[['item_id', 'attribute']], column=['attribute'])
    print("graph emb done!")


    merged_df = merged_df.drop(['item_id', 'new_ID', 'attribute'], axis=1)
    merged_df = merged_df.fillna(0)
    save_data(merged_df.values, root_to + dataset_name+'_feat.dat')
    print("Netease dataset has been done!")

def netease_cxt_pro():
    root = opt_com.root_to + 'netease/' + 'netease_cxt.txt'
    root_to = opt_com.root_to + 'netease/' + 'CXT_netease.dat'
    train_df, cxt_dict = time_propress(root, sep=" ")  # （use,item）：time
    save_data(cxt_dict, root_to)
    print("Netease CXT Preprocess Done!")

def food_preprocess():
    print("===============Processing Food Dataset==================")

    root = opt_com.root + 'food/'
    root_to = opt_com.root_to + 'food/'

    line = 0
    data_1 = pd.read_csv(root + 'interactions_train.csv')
    data_2 = pd.read_csv(root + 'interactions_test.csv')
    df = pd.concat([data_1,data_2],ignore_index=True)
    countU = df.groupby('u')['i'].count().to_dict()
    countP = df.groupby('i')['u'].count().to_dict()
    df['date'] = pd.to_datetime(df['date'])
    # 将 datetime 类型转换为时间戳 (单位为秒)
    df['timestamp'] = df['date'].apply(lambda x: x.timestamp())

    dataset_name = 'food'
    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    for index, l in df.iterrows():
        line += 1
        asin = l['i']
        rev = l['u']
        time = l['timestamp']
        if countU[rev] < opt_com.min_num or countP[asin] < opt_com.min_num : # 数量
            continue
        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = [] # 转换user_id
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum # 这里决定了从1开始标
            itemmap[asin] = itemid
        User[userid].append([time, itemid]) # 每个user id存储的是[item, timestamp]存储的是时间戳{user:[[time, id],[time. id]]}
    # sort reviews in User according to time
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    print(usernum, itemnum)

    f = open(root_to + dataset_name + '_cxt.txt', 'w') # 带时间的
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d %s\n' % (user, i[1], i[0]))
    f.close()

    f = open(root_to + dataset_name + '.txt', 'w') # 存储user-id pair
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d\n' % (user, i[1]))
    f.close()

    # read feature
    df = pd.read_csv(root + 'PP_recipes.csv')
    df = df[['i', 'calorie_level', 'steps_tokens','techniques']]
    df['steps_tokens'] = df['steps_tokens'].apply(ast.literal_eval)
    df['lenstep']  = df['steps_tokens'].apply(lambda x: len(x))
    df = df.drop(['steps_tokens'], axis=1)

    df['techniques'] = df['techniques'].apply(ast.literal_eval)
    list_cols = df['techniques'].apply(pd.Series)
    # 动态创建列名基于列的数量
    list_cols.columns = [f"Column_{i + 1}" for i in range(list_cols.shape[1])]
    df = pd.concat([df, list_cols], axis=1).drop(columns=['techniques'], axis=1)

    # 对category列进行get dummy处理
    dummies = pd.get_dummies(df['calorie_level'], prefix='calorie_level')
    # 将dummy变量合并回原数据框
    result_df = pd.concat([df, dummies], axis=1)
    result_df = result_df.drop(['calorie_level'], axis=1)

    # 基于字典创建一个新的DataFrame
    id_df = pd.DataFrame(list(itemmap.items()), columns=['i', 'new_ID'])
    merged_df = id_df.merge(result_df, on='i', how='left', suffixes=('_dict', ''))
    # merged_df.drop(columns=['i_dict'], inplace=True)
    merged_df = merged_df.fillna(0)
    merged_df = merged_df.sort_values(by='new_ID', ascending=True)
    merged_df = merged_df.drop(['i', 'new_ID'], axis=1)
    print(merged_df.head())
    print(merged_df.columns)

    save_data(merged_df.values, root_to + dataset_name+'_feat.dat')
    print("Food dataset has been done!")

def food_cxt_pro():
    root = opt_com.root_to + 'food/' + 'food_cxt.txt'
    root_to = opt_com.root_to + 'food/' + 'CXT_food.dat'
    train_df, cxt_dict = time_propress(root, sep=" ")  # （use,item）：time
    save_data(cxt_dict, root_to)
    print("CXT Preprocess Done!")

def video_preprocess():
    """交互"""
    print("===============Processing Video Dataset==================")
    root = opt_com.root + 'video/'
    root_to = opt_com.root_to + 'video/'
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0
    # 主要是计数;并且转为txt标准化 user, item, score, timestamp， 完整的文件
    dataset_name = 'Video_Games'
    data_file = root + 'reviews_' + dataset_name + '.json.gz'
    file_name = root_to + 'reviews_' + dataset_name + '.txt'
    f = open(file_name, 'w')
    for l in parse(data_file):
        line += 1
        f.write(" ".join([l['reviewerID'], l['asin'], str(l['overall']), str(l['unixReviewTime'])]) + ' \n')
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        countU[rev] += 1
        countP[asin] += 1
    f.close()

    usermap, usernum = dict(), 0 # ID重编
    itemmap, itemnum = dict(), 0
    User = dict() # {user:[[time, id],[time. id]]}
    for l in parse(root + 'reviews_' + dataset_name + '.json.gz'):
        line += 1
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']
        if countU[rev] < opt_com.min_num or countP[asin] < opt_com.min_num: # 数量
            continue

        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = [] # 转换user_id
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
        User[userid].append([time, itemid]) # 每个user id存储的是[item, timestamp]存储的是时间戳{user:[[time, id],[time. id]]}
    # sort reviews in User according to time
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    print(usernum, itemnum)

    f = open(root_to + dataset_name + '_cxt.txt', 'w') # 有时间信息的
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d %s\n' % (user, i[1], i[0]))
    f.close()

    f = open(root_to + dataset_name + '.txt', 'w') # 没有时间信息的
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d\n' % (user, i[1]))
    f.close()

    #### Reading and writing features
    itemfeat_dict = {}
    counter = 0
    for l in parse(root + 'meta_' + dataset_name + '.json.gz'):
        line += 1
        asin = l['asin']

        title = ""
        if 'description' in l.keys():
            title = l['description']

        price = 0.0
        if 'price' in l.keys():
            price = float(l['price'])

        brand = ""
        if 'brand' in l.keys():
            brand = l['brand']

        categories = l['categories'][0]
        #print(price , "-",brand , "-",categories , "-" )
        if asin in itemmap.keys():
            itemid = itemmap[asin]
            itemfeat_dict[itemid] = [title,price,brand,categories] # {id:[title,price,brand,categories]}
            counter = counter + 1

    features_list = list()
    templist = ["",0.0,"",[]]
    for item_id in range(1,itemnum+1):
        if item_id in itemfeat_dict.keys():
            features_list.append(itemfeat_dict[item_id])
        else:
            features_list.append(templist)

    df = pd.DataFrame(features_list, columns=['title','price','brand','categories'])

    # graph_prepro
    id_item = pd.DataFrame(list(itemmap.items()), columns=['item', 'id'])
    id_item['id'] = id_item['id']
    id_item[['id', 'item']].to_csv(root_to + 'id_item.txt', index=False, header=False)
    video_graph(root_to, df, column=['categories'])


    del df['title']
    df['categoriesstring'] = [' '.join(map(str, l)) for l in df['categories']] # 将category转为字符串
    df=pd.concat([df,df['categoriesstring'].str.get_dummies(sep=' ').add_prefix('cat_').astype('int8')],axis=1) # 一列转为多咧
    del df['categories']
    del df['categoriesstring']
    df=pd.get_dummies(df,dummy_na=True) # 似乎只需要category，我们似乎可以根据category构建我们的图

    print(df.head()) # [price, brand, cat_1, cat_2]
    print(df.dtypes)
    save_data(df.values, root_to + dataset_name+'_feat.dat')
    print("Amazon Video games dataset has been done!")
    return

def video_cxt_pro():
    root = opt_com.root_to + 'video/' + 'Video_Games_cxt.txt'
    root_to = opt_com.root_to + 'video/' + 'CXT_Video_Games.dat'
    train_df, cxt_dict = time_propress(root, sep=" ") # （use,item）：time
    save_data(cxt_dict, root_to)
    print("CXT Preprocess Done!")

def steam_preprocess():
    print("===============Processing Steam Dataset==================")
    dataset_name = 'steam'
    root_to = opt_com.root_to + 'steam/'
    root = opt_com.root + dataset_name

    # countU = defaultdict(lambda: 0)
    # countP = defaultdict(lambda: 0)
    line = 0

    df = pd.read_csv(root + '/recommendations.csv')
    countU = df.groupby('user_id')['app_id'].count().to_dict()
    countP = df.groupby('app_id')['user_id'].count().to_dict()
    df['date'] = pd.to_datetime(df['date'])
    # 将 datetime 类型转换为时间戳 (单位为秒)
    df['ts'] = df['date'].apply(lambda x: x.timestamp())

    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    for index, l in df.iterrows():
        line += 1
        asin = l['app_id']
        rev = l['user_id']
        time = l['ts']
        if countU[rev] < opt_com.min_num or countP[asin] < opt_com.min_num : # 数量
            continue
        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = [] # 转换user_id
        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid
        User[userid].append([time, itemid]) # 每个user id存储的是[item, timestamp]存储的是时间戳{user:[[time, id],[time. id]]}
    # sort reviews in User according to time
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])

    print(usernum, itemnum)

    f = open(root_to + dataset_name + '_cxt.txt', 'w') # 带时间的
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d %s\n' % (user, i[1], i[0]))
    f.close()

    f = open(root_to + dataset_name + '.txt', 'w') # 存储user-id pair
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d\n' % (user, i[1]))
    f.close()

    # read feature
    df = pd.read_csv(root + '/games.csv')
    df = df[['app_id', 'rating','positive_ratio','price_final','discount']]
    # 对category列进行get dummy处理
    dummies = pd.get_dummies(df['positive_ratio'], prefix='positive_ratio')
    result_df = pd.concat([df, dummies], axis=1)
    result_df = result_df.drop(['positive_ratio'], axis=1)

    dummies = pd.get_dummies(result_df['rating'], prefix='rating') # 这就不用后面context做剧了
    result_df = pd.concat([result_df, dummies], axis=1)
    result_df = result_df.drop(['rating'], axis=1)

    # 合
    id_df = pd.DataFrame(list(itemmap.items()), columns=['app_id', 'new_ID'])
    merged_df = id_df.merge(result_df, on='app_id', how='left', suffixes=('_dict', ''))
    merged_df = merged_df.sort_values(by='new_ID', ascending=True)
    merged_df.drop(columns=['app_id', 'new_ID'], inplace=True)
    merged_df = merged_df.fillna(0)

    save_data(merged_df.values, root_to + dataset_name+'_feat.dat')
    print("Steam dataset has been done!")

def steam_cxt_pro():
    root = opt_com.root_to + 'steam/' + 'steam_cxt.txt'
    root_to = opt_com.root_to + 'steam/' + 'CXT_steam.dat'
    train_df, cxt_dict = time_propress(root, sep=" ") # （use,item）：time
    save_data(cxt_dict, root_to)
    print("CXT Preprocess Done!")




def time_propress(filname, sep="\t"):
    col_names = ["user", "item", "ts"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)

    df['ts'] = pd.to_datetime(df['ts'],unit='s')
    df = df.sort_values(by=['ts'])
    if filname.split('/')[-1] == 'netease_cxt.txt':
        print("netease特殊处理 ")
        df['year'], df['month'], df['day'], df['dayofweek'], df['dayofyear'] , df['week'] = zip(*df['ts'].map(lambda x: [x.minute,x.second,x.day,x.dayofweek,x.hour,x.hour]))
        # df['year']-=df['year'].min()
        # df['year']/=df['year'].max()
        df['year'] /= 60
        df['month'] /= 60
        df['day']/=31
        df['dayofweek']/=7
        df['dayofyear']/=365
        df['week']/=24
    else:
        df['year'], df['month'], df['day'], df['dayofweek'], df['dayofyear'] , df['week'] = zip(*df['ts'].map(lambda x: [x.year,x.month,x.day,x.dayofweek,x.dayofyear,x.week]))
        df['year']-=df['year'].min()
        df['year']/=df['year'].max()
        df['month']/=12
        df['day']/=31
        df['dayofweek']/=7
        df['dayofyear']/=365
        df['week']/=4

    DATEINFO = {}
    UsersDict = {}
    for index, row in df.iterrows() :
        userid = int(row['user'])
        itemid = int(row['item'])
        year = row['year']
        month = row['month']
        day = row['day']
        dayofweek = row['dayofweek']
        dayofyear = row['dayofyear']
        week = row['week']
        DATEINFO[(userid,itemid)] = [year, month, day, dayofweek, dayofyear, week] # 获取context数据
    return df, DATEINFO




# 下面是训练测试划分以及data set生成，可以直接import.
def data_partition(dataset_name):
    """读取数据，训练测试划分"""
    if dataset_name == 'video':
        dataset_name = 'Video_Games'
        root = opt_com.root_to + 'video' + '/' + dataset_name + '.txt'
    else:
        root = opt_com.root_to + dataset_name + '/' + dataset_name + '.txt'

    usernum = 0
    itemnum = 0
    User = defaultdict(list) # {A:[1,2,3,4]}
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(root, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum) # 用户编码
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3: 
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    print("Split Done!")
    return [user_train, user_valid, user_test, usernum, itemnum]


def get_ItemData(root, dataset_name):
    """这里获得真实的feature"""
    root = root
    print("Root is ,", root)
    if dataset_name == 'video':
        ItemFeatures = load_data(root + 'Video_Games_feat.dat')
        ItemFeatures = np.vstack((np.zeros(ItemFeatures.shape[1]), ItemFeatures)) # 多了一个为0的
    elif dataset_name == 'steam':
        ItemFeatures = load_data(root + 'steam_feat.dat')
        ItemFeatures = np.array(ItemFeatures[:,1:],dtype=np.float32)
        ItemFeatures = np.vstack((np.zeros(ItemFeatures.shape[1]), ItemFeatures))
    elif dataset_name == 'netease':
        ItemFeatures = load_data(root + 'netease_feat.dat')
        ItemFeatures = np.vstack((np.zeros(ItemFeatures.shape[1]), ItemFeatures))
    elif dataset_name == 'food':
        ItemFeatures = load_data(root + 'food_feat.dat')
        ItemFeatures = np.vstack((np.zeros(ItemFeatures.shape[1]), ItemFeatures))
    return ItemFeatures


# 规范化输入
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def win_pre(user_train):
    new_user_train = defaultdict(list)  # dict
    num = 0
    for i in user_train.keys():
        if len(user_train[i]) <= 2:  # 1肯定是不行的，2也不行，因为拆了也不能用(临界拆分点)
            new_user_train[i].append(user_train[i]) # {'u':[1,2,3],[1,2,3,4]]}
            num +=1
        else:
            for j in range(2, len(user_train[i])):  # 最少也得有2条数据及以上吧
                new_user_train[i].append(user_train[i][:j])
                num +=1
    print("Win preprocess!")
    return new_user_train, num
def win_sample_function(user_train, usernum, itemnum, cxtdict, cxtsize, batch_size, maxlen, result_queue, SEED, new_user_train, forw=False):
    """一条变多条,可变长滑动;model可能用到"""
    # new_user_train = defaultdict(list)  # dict
    # for i in user_train.keys():
    #     if len(user_train[i]) <= 2:  # 1肯定是不行的，2也不行，因为拆了也不能用(临界拆分点)
    #         new_user_train[i].append(user_train[i])
    #     else:
    #         for j in range(2, len(user_train[i])):  # 最少也得有2条数据及以上吧
    #             new_user_train[i].append(user_train[i][:j])
    def sample_for():
        """正向padding,适用于一些RNN-based的基线"""
        user = np.random.randint(1, usernum + 1)
        index = np.random.randint(0, len(new_user_train[user]))
        while len(new_user_train[user][index]) <= 1: user = np.random.randint(1,
                                                                              usernum + 1)  # 对于train中只有一个item的用户，重新采样；【因为其如果有三次的话可能有两次分在了test和valid里，训练只有一次，这样没有办法采集positive的样本】
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)  # 这里就用的序列内监督，没差别；没有使用那种滑动窗口截断的操作。
        neg = np.zeros([maxlen], dtype=np.int32)
        ###CXT
        seqcxt = np.zeros([maxlen, cxtsize], dtype=np.float32)
        poscxt = np.zeros([maxlen, cxtsize], dtype=np.float32)
        negcxt = np.zeros([maxlen, cxtsize], dtype=np.float32)
        ###
        idx = 0
        # nxt = new_user_train[user][index][1]
        ts = set(user_train[user])  # item set
        for i in range(len(new_user_train[user][index][:-1])): # [1,2,3]
            it = new_user_train[user][index][i]
            seq[idx] = it
            nxt = new_user_train[user][index][i+1]  # 其实是下一个item
            pos[idx] = nxt
            neg_i = 0
            if nxt != 0:
                neg_i = random_neq(1, itemnum + 1, ts)
                neg[idx] = neg_i
            ###CXT
            seqcxt[idx] = cxtdict[(user, it)]
            poscxt[idx] = cxtdict[(user, nxt)]  # 下一个item的时间点
            negcxt[idx] = cxtdict[(user, nxt)]  # 相同的时间背景
            ###
            idx += 1
            if idx == maxlen-1: break
        return (np.ones(maxlen) * user, seq, pos, neg, seqcxt, poscxt, negcxt)


    def sample():
        """[1,2,3]-> seq [0,1,2], pos[0,2,3], neg [0,3,4]"""
        user = np.random.randint(1, usernum + 1)
        index = np.random.randint(0, len(new_user_train[user]))  # 随机选择一个长度; 切成了很多块 [[10215, 11619], [10215, 11619, 12436], [10215, 11619, 12436, 12395]]
        while len(new_user_train[user][index]) <= 1: user = np.random.randint(1, usernum + 1) # 对于train中只有一个item的用户，重新采样；【因为其如果有三次的话可能有两次分在了test和valid里，训练只有一次，这样没有办法采集positive的样本】
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32) #  这里就用的序列内监督，没差别；没有使用那种滑动窗口截断的操作。
        neg = np.zeros([maxlen], dtype=np.int32)
        ###CXT
        seqcxt = np.zeros([maxlen,cxtsize], dtype=np.float32)
        poscxt = np.zeros([maxlen,cxtsize], dtype=np.float32)
        negcxt = np.zeros([maxlen,cxtsize], dtype=np.float32)
        ###
        nxt = new_user_train[user][index][-1] # 第n次的item 4
        idx = maxlen - 1
        ts = set(user_train[user]) # item set
        for i in reversed(new_user_train[user][index][:-1]): # 反转,不会踩到-1
            seq[idx] = i
            pos[idx] = nxt # 其实是下一个item
            neg_i = 0
            if nxt != 0:
                neg_i = random_neq(1, itemnum + 1, ts)
                neg[idx] = neg_i
            ###CXT
            seqcxt[idx] = cxtdict[(user,i)]
            poscxt[idx] = cxtdict[(user,nxt)] # 下一个item的时间点
            negcxt[idx] = cxtdict[(user,nxt)] # 相同的时间背景
            ###
            nxt = i
            idx -= 1
            if idx == -1: break
        return (np.ones(maxlen)*user, seq, pos, neg, seqcxt, poscxt, negcxt)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            if forw:
                one_batch.append(sample_for())
            else:
                one_batch.append(sample())
        result_queue.put(zip(*one_batch))


def sample_function(user_train, usernum, itemnum, cxtdict, cxtsize, batch_size, maxlen, result_queue, SEED, new_user_train=None, forw=False):
    """这里是多进程的采样函数，可以定义不同的采样方式"""
    def sample():
        """[1,2,3]-> seq [0,1,2], pos[0,2,3], neg [0,3,4]"""
        user = np.random.randint(1, usernum + 1) # SASRec
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1) # 对于train中只有一个item的用户，重新采样；【因为其如果有三次的话可能有两次分在了test和valid里，训练只有一次，这样没有办法采集positive的样本】
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32) #  这里就用的序列内监督，没差别；没有使用那种滑动窗口截断的操作。
        neg = np.zeros([maxlen], dtype=np.int32)
        ###CXT
        seqcxt = np.zeros([maxlen,cxtsize], dtype=np.float32)
        poscxt = np.zeros([maxlen,cxtsize], dtype=np.float32)
        negcxt = np.zeros([maxlen,cxtsize], dtype=np.float32)
        ###
        nxt = user_train[user][-1] # 第n次的item 4
        idx = maxlen - 1
        ts = set(user_train[user]) # item set
        for i in reversed(user_train[user][:-1]): # 反转,不会踩到-1 [0,0,0,1] [3,2,1]
            seq[idx] = i # 3
            pos[idx] = nxt # 4
            neg_i = 0
            if nxt != 0:
                neg_i = random_neq(1, itemnum + 1, ts)
                neg[idx] = neg_i
            ###CXT
            seqcxt[idx] = cxtdict[(user,i)]
            poscxt[idx] = cxtdict[(user,nxt)] # 下一个item的时间点
            negcxt[idx] = cxtdict[(user,nxt)] # 相同的时间背景
            ###
            nxt = i
            idx -= 1
            if idx == -1: break
        return (np.ones(maxlen)*user, seq, pos, neg, seqcxt, poscxt, negcxt)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    """why using this rather than dataloder"""
    def __init__(self, User, usernum, itemnum, cxtdict, cxtsize, batch_size=64, maxlen=10, n_workers=1, new_user_train=None, forw=False, win=False):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        if win:
            func =win_sample_function
        else:
            func = sample_function
        for i in range(n_workers):
            self.processors.append(
                Process(target=func, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      cxtdict,
                                                      cxtsize,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                    new_user_train,
                                                    forw
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get() # 感觉很像dataloader, 多进程

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def test_data_preprocess(dataset, user_num, cxtdict, cxtsize, negnum=100, opt=None):
    """这里针对其进行1:99负采样"""
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    # 存储negative samples,不然不好弄 1:99测试
    if usernum > 10000:
        print("user num too many!")
        users = random.sample(range(1, usernum + 1), 10000) # 随机选一些测试
    else:
        print("user num", user_num)
        users = range(1, usernum + 1)  # 在1-user num中选？
    # all
    all_infos = []
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue
        seq = np.zeros([opt.maxlen], dtype=np.int32) #[1,2,3,4] ->[1,2,] [3] [4]
        seqcxt = np.zeros([opt.maxlen, cxtsize], dtype=np.int32)
        testitemscxt = list()
        idx = opt.maxlen - 1

        # 下面不是还是存储了historical data，感觉没差别啊
        seq[idx] = valid[u][0]  # 倒数第二位是current interest [XX]
        # Cxt
        seqcxt[idx] = cxtdict[(u, valid[u][0])] # 这个不是有意泄漏了一定的时间信息，就是考量到时间

        idx -= 1
        for i in reversed(train[u]):  # long interest；这样我的test也是增多的。。[1,2]
            seq[idx] = i
            # Cxt
            seqcxt[idx] = cxtdict[(u, i)]

            idx -= 1
            if idx == -1: break #[0,0,0,1,2,3]

        rated = set(train[u])  # [X,X,X]
        rated.add(0)  # (X,X,X,0)，0为padding位置
        item_idx = [test[u][0]]  # test item
        testitemscxt.append(cxtdict[(u, test[u][0])])  # 存储了context [[context]]
        for _ in range(negnum):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1) # 从1开时
            item_idx.append(t)
            testitemscxt.append(cxtdict[(u, test[u][0])])  # [[context], [context]]
        info = (np.ones(opt.maxlen) * u, [seq], item_idx, [seqcxt],
                testitemscxt)  # u_emb (每个时刻都放一个), historical item, target item, historical cxt, test_cxt.

        # 这里可以分类
        '''
        
        if item_idx in tes[0]:
            all_infos.append(info)
        else:
            continue

        '''

        all_infos.append(info)

    return all_infos



def test_data_preprocess_for(dataset, user_num, cxtdict, cxtsize, negnum=100, opt=None):
    """这里针对其进行1:99负采样； 这里不随机取"""
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000) # 随机选一些测试; 1-usernum，取不相同的10000个
    else:
        print('user num,', user_num)
        users = range(1, usernum + 1)  # 在1-user num中选？
    # all
    all_infos = []
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue # 一定需要有训练数据或者测试数据
        seq = np.zeros([opt.maxlen], dtype=np.int32)
        seqcxt = np.zeros([opt.maxlen, cxtsize], dtype=np.int32)
        testitemscxt = list()
        idx = 0 #opt.maxlen - 1
        train[u].append(valid[u][0])
        for i in train[u]:  # long interest
            seq[idx] = i
            # Cxt
            seqcxt[idx] = cxtdict[(u, i)]
            idx += 1
            if idx == opt.maxlen-1: break # [1,2,3,4,0000]

        rated = set(train[u])  # [X,X,X]
        rated.add(0)  # (X,X,X,0)，0为padding位置
        item_idx = [test[u][0]]  # test item
        testitemscxt.append(cxtdict[(u, test[u][0])])  # 存储了context [[context]]
        for _ in range(negnum):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1) # 从1开时
            item_idx.append(t)
            testitemscxt.append(cxtdict[(u, test[u][0])])  # [[context], [context]]
        info = (np.ones(opt.maxlen) * u, [seq], item_idx, [seqcxt],
                testitemscxt)  # u_emb (每个时刻都放一个), historical item, target item, historical cxt, test_cxt.
        all_infos.append(info)

    return all_infos



class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]

def collate_fn(batch):
    users, seqs, item_idxs, seqcxts, testitemscxts = zip(*batch)
    return (users, seqs, item_idxs, seqcxts, testitemscxts)


def get_dataloader(dataset_name, opt, win=False, forw=False):
    """win指的是连续windows, forw指的是正向padding"""
    root = opt_com.root_to + dataset_name + '/'
    dataset = data_partition(dataset_name)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) / opt.batch_size
    print(usernum, '--', itemnum)
    ItemFeatures = None
    UserFeatures = None
    if dataset_name == 'video':
        ItemFeatures = get_ItemData(root, dataset_name)
        CXTDict = load_data(root + 'CXT_Video_Games.dat')
    elif dataset_name == 'steam':
        ItemFeatures = get_ItemData(root, dataset_name)
        CXTDict = load_data(root + 'CXT_steam.dat')
    elif dataset_name == 'netease':
        ItemFeatures = get_ItemData(root, dataset_name)
        CXTDict = load_data(root + 'CXT_netease.dat')
    elif dataset_name == 'food':
        ItemFeatures = get_ItemData(root, dataset_name)
        CXTDict = load_data(root + 'CXT_food.dat')
    else:
        print("No such dataset!")
    print("Item shape", ItemFeatures.shape)

    new_user_train=None

    if win:
        new_user_train, num = win_pre(user_train)
        num_batch = num / opt.batch_size
    # 以上没有问题
    sampler = WarpSampler(user_train, usernum, itemnum, CXTDict, opt.cxt_size, batch_size=opt.batch_size,
                          maxlen=opt.maxlen, n_workers=3, new_user_train=new_user_train, forw=forw, win=win)

    if forw:
        test_data = test_data_preprocess_for(dataset, usernum, CXTDict, opt.cxt_size, negnum=opt_com.negnum, opt=opt)
    else:
        test_data = test_data_preprocess(dataset, usernum, CXTDict, opt.cxt_size, negnum=opt_com.negnum, opt=opt) # lis; 那这个应该不影响才对啊；
    test_sampler = DataLoader(CustomDataset(test_data), batch_size=opt_com.tes_batch, shuffle=False, collate_fn=collate_fn, drop_last=True) # 不要搞的很大
    # opt.batch_size // 2
    return sampler, test_sampler, num_batch, ItemFeatures, usernum, itemnum


class SpecArgs:
    batch_size = 64
    cxt_size = 6 # food: 6 ; 这里的cxt size应该都是6，因为其实是6个时间特征
    maxlen = 10 # 这里的max_len包括最后的target

opt_xx = SpecArgs()

if __name__ == '__main__':
    print("test")
    data_preprocess('netease')
    # user_train, user_valid, user_test, usernum, itemnum = data_partition('video') # 测试
    # get_dataloader('netease', opt_xx) # netease统计年，月，日没啥意义
    # root = opt_com.root_to + 'steam/'
    # Item_features = get_ItemData(root, 'steam')
    # trans = np.array(Item_features[:,1:],dtype=np.float32)
    # print('success')
