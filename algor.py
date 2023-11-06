import os
import warnings
import heapq

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from dataset_2 import Grapth_Datase

from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, BatchNorm
from torch_geometric.nn import global_mean_pool
from torch.nn import ModuleList
from evaluate import roc,score,recall,precision


from torch_geometric.nn import GAE,GATConv,VGAE,GraphSAGE,GENConv,GINConv,GMMConv
from torch_geometric.nn.conv import GINConv,GMMConv,GATConv,GCN2Conv,GeneralConv,EGConv,SAGEConv,EdgeConv
from models.pytorch_geometric.pna import PNAConvSimple



class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(num_node_features, 64)
        self.bn_1 = BatchNorm(64)
        self.conv2 = GraphConv(64, 32)
        self.conv3 = GraphConv(32, 20)
        self.bn_2 = BatchNorm(20)
        self.lin = Linear(20, 11)
        self.softmax = nn.Softmax()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. 获得节点嵌入
        x = self.conv1(x, edge_index)
        x = self.bn_1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn_2(x)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. 分类器
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        #x = self.softmax(x)

        return x


class GAT_NET(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=4):
        super(GAT_NET, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=4)  # 定义GAT层，使用多头注意力机制
        self.gat2 = GATConv(hidden * heads, 32, heads=4,dropout=0.2)  # 因为多头注意力是将向量拼接，所以维度乘以头数。
        self.gat3 = GATConv(32 * 4, 20, heads=1)


        self.bn_1 = BatchNorm(64 * 4)
        self.bn_2 = BatchNorm(32 * 4)
        self.bn_3 = BatchNorm(20)

        self.lin = Linear(20, classes)
        self.softmax = nn.Softmax()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gat1(x, edge_index)
        x = self.bn_1(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)
        x = self.bn_2(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.gat3(x, edge_index)
        x = self.bn_3(x)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. 分类器
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        x = self.softmax(x)

        return x

class GraphSAGE_NET(torch.nn.Module):

    def __init__(self, num_node_features):
        super(GraphSAGE_NET, self).__init__()
        torch.manual_seed(12345)
        self.sage1 = SAGEConv(num_node_features, 64)
        self.bn_1 = BatchNorm(64)
        self.sage2 = SAGEConv(64, 32)
        self.sage3 = SAGEConv(32, 20)
        self.bn_2 = BatchNorm(20)
        self.lin = Linear(20, 11)
        self.softmax = nn.Softmax()

    def forward(self, data,z):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. 获得节点嵌入
        x = self.sage1(x, edge_index)
        x = self.bn_1(x)
        x = x.relu()
        x = self.sage2(x, edge_index)
        x = x.relu()
        x = self.sage3(x, edge_index)
        x = self.bn_2(x)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. 分类器
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        x = self.softmax(x)

        return x

class PNA_Net(torch.nn.Module):
    def __init__(self,deg,num_node_features):#加了个deg
        super(PNA_Net, self).__init__()
        #self.node_emb = AtomEncoder(emb_dim=80)     #咋整咋整咋整

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.conv0 = PNAConvSimple(in_channels=num_node_features, out_channels=80, aggregators=aggregators,
                             scalers=scalers, deg=deg, post_layers=1)
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        for _ in range(4):
            conv = PNAConvSimple(in_channels=80, out_channels=80, aggregators=aggregators,
                                 scalers=scalers, deg=deg, post_layers=1)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(80))

        self.mlp = Sequential(Linear(80, 40), ReLU(), Linear(40, 20), ReLU(), Linear(20, 11)) ##10or11
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.lin = Linear(80, 11)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        #x = self.node_emb(x)
        x = self.conv0(x, edge_index, edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            x = h + x  # residual#
            x = F.dropout(x, 0.3, training=self.training)

        x = global_mean_pool(x, batch)

        #x = self.mlp(x)
        x = self.lin(x)
        x = self.softmax(x)

        return x



class PNA(torch.nn.Module):

    def __init__(self,deg, num_node_features):
        super(PNA, self).__init__()
        torch.manual_seed(12345)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']


        self.pna1 = PNAConvSimple(in_channels=num_node_features, out_channels=64, aggregators=aggregators,
                                   scalers=scalers, deg=deg, post_layers=1)
        self.bn_1 = BatchNorm(64)
        self.pna2 = PNAConvSimple(in_channels=64, out_channels=32, aggregators=aggregators,
                                   scalers=scalers, deg=deg, post_layers=1)
        self.pna3 = PNAConvSimple(in_channels=32, out_channels=20, aggregators=aggregators,
                                   scalers=scalers, deg=deg, post_layers=1)
        self.bn_2 = BatchNorm(20)
        self.lin = Linear(20, 11)
        self.softmax = nn.Softmax()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. 获得节点嵌入
        x = self.pna1(x, edge_index)
        x = self.bn_1(x)
        x = x.relu()
        x = self.pna2(x, edge_index)
        x = x.relu()
        x = self.pna3(x, edge_index)
        x = self.bn_2(x)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. 分类器
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        x = self.softmax(x)

        return x