# -*- coding: utf-8 -*-
# @Time    : 2023/7/23 18:04
# @Author  : ljj
# @File    : data_process_1_2.py

import warnings
import os
import torch
import math
from torch_geometric.data import Data
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
edge_index = [
    [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 11, 11,
     12, 12, 12, 12, 13, 13, 14, 14,
     14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23],
    [3, 4, 4, 5, 5, 6, 7, 0, 0, 1, 7, 8, 1, 2, 8, 9, 2, 9, 3, 4, 10, 11, 4, 5, 11, 12, 5, 6, 12, 13, 7, 14, 7, 8, 14,
     15, 8, 9, 15, 16, 9, 16, 10, 11,
     17, 18, 11, 12, 18, 19, 12, 13, 19, 20, 14, 21, 14, 15, 21, 22, 15, 16, 22, 23, 16, 23, 17, 18, 18, 19, 19, 20]]

node_index = {
    "01": "0",
    "12": "1",
    "23": "2",
    "04": "3",
    "15": "4",
    "26": "5",
    "37": "6",
    "45": "7",
    "56": "8",
    "67": "9",
    "48": "10",
    "59": "11",
    "6a": "12",
    "7b": "13",
    "89": "14",
    "9a": "15",
    "ab": "16",
    "8c": "17",
    "9d": "18",
    "ae": "19",
    "bf": "20",
    "cd": "21",
    "de": "22",
    "ef": "23"
}

dct = ['01', '04', '12', '15', '23', '26', '37', '45', '48', '56', '59', '67', '6a', '7b', '89', '8c', '9a', '9d', 'ab',
       'ae', 'bf', 'cd', 'de', 'ef']

pos2 = [(1, 7), (3, 7), (5, 7), (0, 6), (2, 6), (4, 6), (6, 6), (1, 5), (3, 5), (5, 5), (0, 4), (2, 4), (4, 4), (6, 4),
        (1, 3), (3, 3), (5, 3), (0, 2), (2, 2), (4, 2), (6, 2), (1, 1), (3, 1), (5, 1)]


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def dataset_mkdir(root_name_list):
    root_name = ""
    for i in range(len(root_name_list)):
        root_name = root_name + root_name_list[i]
        if not os.path.exists(root_name):
            os.mkdir(root_name)
        root_name = root_name + "/"
    print(root_name)


gragh_root = []
root_concentrations = []
root = "data_nor"
root_speeds = [root + "/" + i for i in os.listdir(root)]
for root_speed in root_speeds:
    root_concentrations.extend([root_speed + "/" + i for i in os.listdir(root_speed)])
for root_concentration in root_concentrations:
    gragh_root.extend([root_concentration + "/" + i for i in os.listdir(root_concentration)])

num = 0
for data_path in gragh_root:
    ar_load = np.load(data_path)
    x = torch.empty(24, 3)
    mean_R = 0
    stand_R = 0
    mean_phase = 0
    stand_phase = 0
    flag = 1

    for i in range(ar_load.shape[0]):
        ar_load[i][0] = node_index[ar_load[i][0]]
        ar_load[i][1] = node_index[ar_load[i][1]]
    ar_load = ar_load.astype(np.float64)
    for i in range(ar_load.shape[0]):
        k = 0
        d_sum = 0
        for m in range(24):
            d_sum += distance(pos2[i], pos2[m]) / 2
        d_mean = d_sum / 24
        for j in range(1, ar_load.shape[1]):
            if j % 3 == 0:
                try:
                    d = 1
                    # d = distance(pos2[i], pos2[int(node_index[dct[k]])]) / 2 + 1
                    # d = d / d_mean
                    # ar_load[i][j] = math.log10(ar_load[i][j])
                    # ar_load[i][j + 1] = ar_load[i][j + 1] % 360
                    # ar_load[i][j + 1] = math.log(4, ar_load[i][j + 1])
                    ar_load[i][j] = math.log10((ar_load[i][j] / d) + 800000)
                    ar_load[i][j + 1] = ar_load[i][j + 1] % 360
                    # print(ar_load[i][j + 3]<0)
                    ar_load[i][j + 1] = math.log(4, ar_load[i][j + 1] + 2)
                    k = k + 1

                # if j % 3 == 0:
                #     try:
                #         ar_load[i][j] = math.log10(ar_load[i][j])
                #         ar_load[i][j + 1] = ar_load[i][j + 1] % 360
                #         ar_load[i][j + 1] = math.log(4, ar_load[i][j + 1])
                except:
                    print("except")
                    flag = 0
                    num = num + 1

    if flag == 1:
        mu = np.mean(ar_load, axis=0)
        sigma = np.std(ar_load, axis=0)
        for i in range(1, ar_load.shape[1]):
            if i % 3 == 0:
                for j in range(ar_load.shape[0]):
                    ar_load[j][i] = (ar_load[j][i] - mu[i]) / sigma[i]
                    ar_load[j][i + 1] = (ar_load[j][i + 1] - mu[i + 1]) / sigma[i + 1]
                    # ar_load[j][i + 2] = (ar_load[j][i + 2] - mu[i + 2]) / sigma[i + 2]
                    # ar_load[j][i + 3] = (ar_load[j][i + 3] - mu[i + 3]) / sigma[i + 3]

            # if i % 3 == 0:
            #     for j in range(ar_load.shape[0]):
            #         ar_load[j][i] = (ar_load[j][i] - mu[i]) / sigma[i]
            #         ar_load[j][i+1] = (ar_load[j][i+1] - mu[i+1]) / sigma[i+1]

        ar_load = ar_load[np.argsort(ar_load[:, 1])][:, 2:]
        edge = torch.tensor(edge_index, dtype=torch.long)
        ar_load = torch.tensor(ar_load, dtype=torch.float)
        data = Data(x=ar_load, edge_index=edge)
        dst_path_list = data_path.replace("data_nor", "data_graph_nor").split("/")
        graph_path = dst_path_list[:-1]
        dataset_mkdir(graph_path)
        scr_name = dst_path_list[-1]
        graph_name = scr_name.split(".npy")[0]
        graph_path.append(graph_name)
        graph_path = "/".join(graph_path)
        torch.save(data, graph_path)
        print(graph_name)
        print("123")

print(num)
