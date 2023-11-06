# -*- coding: utf-8 -*-
# @Time    : 2023/7/24 21:28
# @Author  : ljj
# @File    : dataset_1.py
import os
import warnings
import torch

warnings.filterwarnings('ignore')


class Grapth_Datase():
    def __init__(self,root):
        self.file_list = []
        labels = []
        gragh_root = []
        root_concentrations = []
        root_speeds = [root + "/" + i for i in os.listdir(root)]
        for root_speed in root_speeds:
            root_concentrations.extend([root_speed + "/" + i for i in os.listdir(root_speed)])
        for root_concentration in root_concentrations:
            gragh_root.extend([root_concentration + "/" + i for i in os.listdir(root_concentration)])
        for i in range(len(gragh_root)):
            self.file_list.append(gragh_root[i])
            label = int(gragh_root[i].split("/")[2])
            labels.append(label)
        ones = torch.sparse.torch.eye(11)
        self.label = ones.index_select(0, torch.LongTensor(labels))
        print("123")
        # self.label = labels


    def len_(self):
        """
        返回数据集中的样本数
        :return:
        """
        return len(self.file_list)

    def getset(self):
        """
        LSTM的输入为(sanmple_num, seq_len, input_size)
        conv1d的输入为(batch,inputchannel,len)
        :param index:
        :return:
        """
        data_list = []
        for index in range(len(self.file_list)):
            graph = torch.load(self.file_list[index])
            graph.y = self.label[index]
            data_list.append(graph)
        return data_list
        # return data_list, self.label

