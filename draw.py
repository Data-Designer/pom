# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : draw.py
# Time       ：9/8/2024 2:59 pm
# Author     ：xxxx
# version    ：python 
# Description： 绘图
"""
import numpy as np
import matplotlib.pyplot as plt


def bar_plot(x,y,x_name,color,marker,width, title, ylim, y_label, save_name=None):
    """ Plug-in LLM, Ablation Study"""
    fig = plt.figure(figsize=(8,5))
    fontdict = {'family': 'Times New Roman',
                'style': 'italic',
                'color': 'black',
                'weight': 'normal',
                'size': 15}
    ax1 = fig.add_subplot(111)
    bars_1 = ax1.bar(x, y, color=color, edgecolor="k", width=width)
    for bar, mark, name in zip(bars_1, marker, x_name):
        bar.set_hatch(mark)
        bar.set_label(name)
    ax1.set_xticks(x)  # Set x-ticks at the positions of the bars
    ax1.set_xticklabels(x_name, fontdict=fontdict)
    plt.ylim(ylim[0], ylim[1]) # y 范围 50-85, 50-100
    ax1.set_ylabel(y_label, fontdict)
    plt.title(title, fontdict=fontdict)
    plt.savefig("./img/{}.pdf".format(save_name), dpi=600, format='pdf')
    # plt.show()



######## different rational aspects bar########
def create_bar_chart_with_markers(data, x_labels, colors, markers, y_label=None, title=None, y_lim=None, save_name=None):
    """
    diverse KG scenario
    """
    fontdict = {'family': 'Times New Roman',
                'style': 'italic',
                'color': 'black',
                'weight': 'normal',
                'size': 14}
    x = np.arange(len(x_labels))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(5, 5))

    for i, (attribute, measurement) in enumerate(data.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i])

        # Adding markers to the bars
        for rect in rects:
            height = rect.get_height()
            rect.set_hatch(markers[multiplier])
        #         ax.bar_label(rects, padding=3) # 加数字
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label, fontdict=fontdict)
    ax.set_title(title,fontdict=fontdict)
    ax.set_xticks(x + 1 * width)
    ax.set_xticklabels(x_labels, fontdict=fontdict)
    ax.legend(loc='upper left')  # ncol=3
    ax.set_ylim(y_lim[0], y_lim[1])

    plt.savefig("./img/{}.pdf".format(save_name), dpi=600, format='pdf')
    plt.show()

def custom_line(x,x_name, labels, y, y2, y3, title, save_name):
    """超参数"""
    fig = plt.figure(figsize=(5, 5))
    fontdict = {'family': 'Times New Roman',
                'style': 'italic',
                'color': 'black',
                'weight': 'normal',
                'size': 15}
    line_width = 2

    # 插值平滑
    # from scipy.interpolate import make_interp_spline
    # model_hit = make_interp_spline(x, y)
    # x = np.linspace(0, 120, 1000) # alpha修改
    # y = model_hit(x)
    # y = smooth(y)
    # yest = lowess(y, x, frac=1. / 3.)[:, 1]

    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Value', fontdict)
    ax1.set_xlabel(x_name, fontdict)
    ax1.plot(x,y,"--",linewidth=line_width,color='#0082807F',label=labels[0],marker='^')
    ax1.plot(x,y2,"--",linewidth=line_width,color='#5F559B7F',label=labels[1],marker='*')
    ax2 = ax1.twinx()
    ax2.plot(x,y3,"--",linewidth=line_width,color='#BB00217F',label=labels[2],marker='v')
    ax2.legend(fontsize=fontdict['size'], loc='center right')
    ax1.legend(fontsize=fontdict['size'],loc='center left')
    plt.title(title, fontdict=fontdict)
    # plt.show()
    plt.savefig("./img/{}.pdf".format(save_name), dpi=600, format='pdf')


if __name__ == '__main__':
    # ablation study
    # *+-./ OX\ox |
    marker = ['o',  '+', '-','.',  'X', '|','O', '/','*']
    marker = ['o',  '+', '*','.',  'X', '|','O', '/','*']

    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    y_label = "Value"
    #
    # data = {
    # "-NT": 0.2900, "-NO": 0.2931, "-NC": 0.2942, "-NG": 0.2817, "-NGI": 0.2916, "-NU": 0.3050, "-NS":0.2891, "-NAUX":0.3012 , "RERec":0.3111,
    # }
    #
    # titile = "Hit@5"
    # bottom, top = min(data.values())*0.85, max(data.values())*1.02
    # ylim = [bottom, top]
    # bar_plot(list(data.keys()), list(data.values()), list(data.keys()), color, marker, 1, titile, ylim, y_label, save_name='aba-hit-5')
    #
    # data = {
    #     "-NT": 0.2800, "-NO": 0.2836, "-NC": 0.2851, "-NG": 0.2731, "-NGI": 0.2888, "-NU": 0.2900, "-NS": 0.2763, "-NAUX": 0.2881, "RERec": 0.2967,
    # }
    # titile = "NDCG@5"
    # bottom, top = min(data.values()) * 0.85, max(data.values()) * 1.02
    # ylim = [bottom, top]
    # bar_plot(list(data.keys()), list(data.values()), list(data.keys()), color, marker, 1, titile, ylim,
    #                     y_label, save_name='aba-ndcg-5')
    #
    # data = {
    #     "-NT": 0.1350, "-NO": 0.1385, "-NC": 0.1380, "-NG": 0.1263, "-NGI": 0.1396, "-NU": 0.14001, "-NS": 0.1294, "-NAUX": 0.1388, "RERec": 0.1481,
    # }
    # titile = "MRR@5"
    # bottom, top = min(data.values()) * 0.85, max(data.values()) * 1.02
    # ylim = [bottom, top]
    # bar_plot(list(data.keys()), list(data.values()), list(data.keys()), color, marker, 1, titile, ylim,
    #                     y_label, save_name='aba-mrr-5')

    #

    # data = {
    #     'DUPN': (0.3287, 0.2987, 0.2900, 0.2675, 0.3107, 0.2234),
    #     'CACAR': (0.3210, 0.2930, 0.2876,0.2575, 0.3127, 0.2209),
    #     'RERec': (0.3207, 0.3014, 0.3004, 0.2851, 0.3128, 0.2571),
    # }
    # x_labels = ["Attack", "Defense", "HP", "Speed", "Charm", "Magic"]
    # colors = color[:6]
    # markers = marker[:6]
    # y_label = 'Value'
    # title = "Hit@5"
    # create_bar_chart_with_markers(data, x_labels, colors, markers,y_label, title, y_lim=[0.2209*0.9, 0.3287*1.1], save_name='rata-hit-5')
    #
    #
    #
    # data = {
    #     'DUPN': (0.3203, 0.2934, 0.2843, 0.2667, 0.2988, 0.2167),  # 1481
    #     'CACAR': (0.3158, 0.2900, 0.2837,0.2567, 0.3006, 0.2163), # 2967
    #     'RERec': (0.3145, 0.2984, 0.2946, 0.2768, 0.3110, 0.2667),  # 3111
    #
    # }
    # x_labels = ["Attack", "Defense", "HP", "Speed", "Charm", "Magic"]
    # colors = color[:6]
    # markers = marker[:6]
    # y_label = 'Value'
    # title = "NDCG@5"
    # create_bar_chart_with_markers(data, x_labels, colors, markers,y_label, title, y_lim=[0.2163*0.9, 0.3203*1.1], save_name='rata-mrr-5')
    #
    # data = {
    #     'DUPN': (0.1611, 0.1436, 0.1337, 0.1201, 0.1404, 0.0956),
    #     'CACAR': (0.1607, 0.1430, 0.1277,0.1011, 0.1434, 0.0934),
    #     'RERec': (0.1581, 0.1446, 0.1521, 0.1211, 0.1455, 0.1011),
    #
    # }
    # x_labels = ["Attack", "Defense", "HP", "Speed", "Charm", "Magic"]
    # colors = color[:6]
    # markers = marker[:6]
    # y_label = 'Value'
    # title = "MRR@5"
    # create_bar_chart_with_markers(data, x_labels, colors, markers,y_label, title, y_lim=[0.0934*0.9, 0.1611*1.1], save_name='rata-ndcg-5')
    #
    #
    # # correlation analysis
    #
    #
    # data = {
    #     'Rational': (0.1645, 0.1726,0.1896),
    #     'Emotional': (0.0482, 0.1030,0.0112),
    #     'Purchase': (0.1373, 0.1375, 0.1589),
    # }
    # x_labels = ["Pearson", "Cosine", "Spearman"]
    # colors = color[:3]
    # markers = marker[:3]
    # y_label = 'Value'
    # title = "Correlation analysis"
    # create_bar_chart_with_markers(data, x_labels, colors, markers,y_label, title, y_lim=[0.0112*0.9, 0.1896*1.1], save_name='corr')
    #
    #
    # # swap two sequence
    # data = {
    #     'Hit@5': (0.0855, 0.3111),
    #     'NDCG@5': (0.0404, 0.2967),
    #     'MRR@5': (0.0270, 0.1481),
    # }
    # x_labels = ["Swap", "No Swap"]
    # colors = color[:3]
    # markers = marker[:3]
    # y_label = 'Value'
    # title = "Performance of swapping two sequence."
    # create_bar_chart_with_markers(data, x_labels, colors, markers, y_label, title, y_lim=[0.0270*0.9, 0.3111*1.1], save_name='swap-5')
# '''
# val_loss = 547.2396,val_AUC  = 0.0000, Hit@10  = 0.0855, NDCG@10  = 0.0404, MRR@10  = 0.0270,
# Rank 1: 0.010799999999999981 0.010799999999999981 0.010799999999999981
# Rank 5: 0.043700000000000266 0.027030673518914687 0.021605000000000027
# Rank 15: 0.12289999999999969 0.05025538043449484 0.029947650960151024
# Rank 20: 0.16799999999999898 0.06090123136656845 0.032477685359875864
# '''

    # # # Number of motives
    # Hit = {"2": 0.3111, "4": 0.3013, "6": 0.3054, "8": 0.2991}
    # NDCG = {"2": 0.2967, "4": 0.2939, "6": 0.2950, "8": 0.2934}
    # MRR = {"2": 0.1481, "4": 0.1441, "6": 0.1444, "8": 0.1442}
    # x, labels = list(Hit.keys()), ['Hit@5', 'NDCG@5', 'MRR@5']
    # y, y2, y3= list(Hit.values()),list(NDCG.values()),  list(MRR.values())
    # custom_line(x,'Motivation Number' ,labels, y, y2, y3, title='Number of Motives.', save_name='motive-5')
    #
    #
    # # # Number of embs,
    # Hit = {"16": 0.1600, "32": 0.3111, "48": 0.3001, "64": 0.2987}
    # NDCG = {"16": 0.1200, "32": 0.2967, "48": 0.2900, "64": 0.2863}
    # MRR = {"16": 0.0500, "32": 0.1481, "48": 0.1476, "64": 0.1330}
    # x, labels = list(Hit.keys()), ['Hit@5', 'NDCG@5', 'MRR@5']
    # y, y2, y3= list(Hit.values()),list(NDCG.values()),  list(MRR.values())
    # custom_line(x,'Embedding Size' ,labels, y, y2, y3, title='Embedding Size.', save_name='emb-5')
    #
    # # # Number of alpha,
    # Hit = {"0.2": 0.2763, "0.4": 0.3097, "0.5": 0.3111, "0.8": 0.3008}
    # NDCG = {"0.2": 0.2500, "0.4": 0.2879, "0.5": 0.2967, "0.8": 0.2854}
    # MRR = {"0.2": 0.1233, "0.4": 0.1431, "0.5": 0.1481, "0.8": 0.1422}
    # x, labels = list(Hit.keys()), ['Hit@5', 'NDCG@5', 'MRR@5']
    # y, y2, y3= list(Hit.values()),list(NDCG.values()),  list(MRR.values())
    # custom_line(x,'Alpha Value' ,labels, y, y2, y3, title='Auxiliary Task Weight.', save_name='alpha-5')
    #


#
# val_loss = 547.2358,val_AUC  = 0.0000, Hit@10  = 0.3054, NDCG@10  = 0.2050, MRR@10  = 0.1744,
# Rank 1: 0.12729999999999955 0.12729999999999955 0.12729999999999955
# Rank 5: 0.2320999999999991 0.18143052234397652 0.16472333333333314
# Rank 15: 0.36030000000000023 0.2194226335061004 0.17860675491175476
# Rank 20: 0.4064000000000002 0.23028827811387437 0.18118078354952566

    # calcute freq & repurchase



