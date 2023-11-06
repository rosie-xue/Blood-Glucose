import os
import warnings

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch_geometric.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from dataset_2_000 import Grapth_Datase
# from dataset_2_500 import Grapth_Datase
from dataset_2 import Grapth_Datase
from torch.nn import Linear
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import global_mean_pool
from evaluate import score, recall, precision
from algor import GCN,GAT_NET,GraphSAGE_NET#,PNA,PNA_Net
from torch_geometric.utils import degree
from try_edge import kmeanss, switch, drop_n, switch_n, switch_n2,switch_n3

from models.pytorch_geometric.pna import PNAConvSimple, PNAConv

warnings.filterwarnings('ignore')
# plt.rcParams['font.sans-serif'] = 'Times New Roman'

name = 'PNA_x'


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mkdir(dir):
    folder = os.path.exists(dir)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(dir)  # makedirs 创建文件时如果路径不存在会创建这个路径
    else:
        print("---  There is this folder!  ---")


def mask(x, p):
    return torch.tensor(np.random.binomial(n=1, p=1 - p, size=x.size()[0])).reshape(-1, 1).to('cuda')


class PNA(torch.nn.Module):

    def __init__(self, deg, num_node_features):
        super(PNA, self).__init__()
        torch.manual_seed(12345)

        aggregators = ['mean', 'min', 'max', 'std'] #sum,mean,min,max,var,std
        #aggregators = ['std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.pna0 = PNAConvSimple(in_channels=num_node_features, out_channels=96, aggregators=aggregators,
                                  scalers=scalers, deg=deg, post_layers=1)
        self.bn_0 = BatchNorm(96)
        self.pna1 = PNAConvSimple(in_channels=96, out_channels=64, aggregators=aggregators,
                                  scalers=scalers, deg=deg, post_layers=1)
        self.bn_1 = BatchNorm(64)
        self.pna2 = PNAConvSimple(in_channels=64, out_channels=32, aggregators=aggregators,
                                  scalers=scalers, deg=deg, post_layers=1)
        self.pna3 = PNAConvSimple(in_channels=32, out_channels=20, aggregators=aggregators,
                                  scalers=scalers, deg=deg, post_layers=1)

        self.bn_2 = BatchNorm(20)
        self.lin = Linear(20, 11)
        self.softmax = nn.Softmax()
        #self.relu = nn.ReLU()


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        #1. Graph embedding
        x = self.pna0(x, edge_index)
        x = self.bn_0(x)
        x = x.relu()
        x = self.pna1(x, edge_index)
        x = self.bn_1(x)
        x = x.relu()
        x = self.pna2(x, edge_index)
        x = x.relu()
        x = self.pna3(x, edge_index)
        x = self.bn_2(x)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        z = x

        # 3. classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        x = self.softmax(x)

        return x, z


def training(train_set, test_set, lr, n_epoch, device, batchsize, num_id, output_list, yy, count):
    deg = torch.zeros(10, dtype=torch.long)
    for data in train_set:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    n = train_set[0].num_node_features
    #model = GCN(n)
    # model = GAT_NET(n, 64, 11, heads=4)
    # model = GraphSAGE_NET(n)  # 定义GraphSAGE
    # model = PNA_Net(deg,n)

    model = PNA(deg, n)
    model = model.to(device)
    model.train()  # 将model的模式设为train，这样optimizer就可以更新model的参数
    loss_function = nn.CrossEntropyLoss()  # 定义损失函數，这里我们使用binary cross entropy loss

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # 将模型的参数传给optimizer，并赋予适当的learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5, last_epoch=-1)  # 学习率的动态调整
    trainloader = DataLoader(dataset=train_set, batch_size=batchsize, shuffle=True)
    testloader = DataLoader(dataset=test_set, batch_size=batchsize, shuffle=True)

    acc_train_list = []
    loss_train_list = []

    acc_test_list = []
    precision_test_list = []
    recall_test_list = []
    loss_test_list = []

    conf_matrix_train_list = torch.zeros(2, 2)
    conf_matrix_test_list = torch.zeros(2, 2)

    for epoch in range(n_epoch):
        # print("当前学习率：  ", optimizer.state_dict()['param_groups'][0]['lr'])
        # print("----------------------train----------------------------")
        # print("----------------------train----------------------------")
        # print("----------------------train----------------------------")
        model.train()
        total_loss, total_acc = 0, 0
        acc_train, loss_train = 0, 0

        loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=True)
        conf_matrix = torch.zeros(2, 2)
        conf_matrix = conf_matrix.to(device)

        for i, batch in loop:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs, cut = model(batch)
            y = batch.y.reshape(-1, 11)
            loss = loss_function(outputs, y)  # 计算此时模型的training loss
            loss.backward()  # 算loss的gradient
            optimizer.step()  # 更新训练模型的參數
            correct = score(outputs.to('cpu'), y.to("cpu"))  # 计算此时模型的training accuracy

            total_acc += correct
            total_loss += loss.item()

            loss_train = total_loss / (i + 1)
            acc_train = total_acc / (i + 1)

            loop.set_description(f'Epoch_train [{epoch + 1}/{n_epoch}]')
            loop.set_postfix(loss=loss_train, acc=acc_train)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        scheduler.step()

        # print("---------------------test----------------------------")
        # print("---------------------test----------------------------")
        # print("---------------------test----------------------------")
        model.eval()
        total_loss, total_acc, total_pre, total_rec = 0, 0, 0, 0
        loss_test, acc_test, pre_test, rec_test = 0, 0, 0, 0
        loop = tqdm(enumerate(testloader), total=len(testloader), leave=True)

        meanpool = np.array([[0] * 20])
        true_list = []

        for i, batch in loop:
            batch = batch.to(device)
            outputs, cut = model(batch)
            y = batch.y.reshape(-1, 11)

            # evaluation
            loss = loss_function(outputs, y)  # 计算此时模型的training loss
            correct = score(outputs.to('cpu'), y.to("cpu"))  # 计算此时模型的training accuracy
            pre = precision(outputs.to('cpu'), y.to("cpu"))
            rec = recall(outputs.to('cpu'), y.to("cpu"))

            total_acc += correct
            total_loss += loss.item()
            total_pre += pre
            total_rec += rec

            loss_test = total_loss / (i + 1)
            acc_test = total_acc / (i + 1)
            pre_test = total_pre / (i + 1)
            rec_test = total_rec / (i + 1)

            loop.set_description(f'Epoch_test [{epoch + 1}/{n_epoch}]')
            loop.set_postfix(loss=loss_test, acc=acc_test, pre=pre_test, recall=rec_test)

            if epoch == 49:
                meanpool = np.concatenate([meanpool, cut.to('cpu').detach().numpy()], axis=0)
                tru = y.to('cpu').detach().numpy()
                out = outputs.to('cpu').detach().numpy()
                for i in range(tru.shape[0]):
                    true_list.append(tru[i].argmax())
                    yy.append(tru[i])
                    output_list.append(out[i])

        acc_test_list.append(acc_test)
        loss_test_list.append(loss_test)
        precision_test_list.append(pre_test)
        recall_test_list.append(rec_test)

        # draw PCA plot

        if epoch == 49:
            meanpool = meanpool[1:, :]
            pca = PCA(n_components=3)
            reduced = pca.fit_transform(meanpool)
            plt.figure()
            plt.rcParams['savefig.dpi'] = 500
            ax = plt.axes(projection='3d')
            b = ax.scatter3D(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=true_list, s=10)
            ax.w_xaxis.set_pane_color('white')
            ax.w_yaxis.set_pane_color('white')
            ax.w_zaxis.set_pane_color('white')
            plt.legend(handles=b.legend_elements()[0], labels=['{}g/L'.format(i) for i in range(11)],
                       bbox_to_anchor=(1.3, 1), borderaxespad=1, fontsize=9)
            # plt.savefig('plots/{}/PCA3D_{}.jpg'.format(name, num_id))
            # plt.show()

    mkdir("results/{}/{}".format(name, num_id))
    torch.save(model.to("cpu"), "results/{}/{}/gcn{}.pth".format(name, num_id, count))
    torch.save(loss_train_list, "results/{}/{}/loss_train_list{}.pt".format(name, num_id, count))
    torch.save(acc_train_list, "results/{}/{}/acc_train_list{}.pt".format(name, num_id, count))
    torch.save(loss_test_list, "results/{}/{}/loss_test_list{}.pt".format(name, num_id, count))
    torch.save(acc_test_list, "results/{}/{}/acc_test_list{}.pt".format(name, num_id, count))
    torch.save(precision_test_list, "results/{}/{}/precision_test_list{}.pt".format(name, num_id, count))
    torch.save(recall_test_list, "results/{}/{}/recall_test_list{}.pt".format(name, num_id, count))


def train_fun_1(count):
    graph_data_set = Grapth_Datase("data_graph_x").getset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 60
    lr = 0.5
    batchsize = 64
    kf = KFold(n_splits=5, shuffle=True)
    num_id = 0

    output_list = []
    yy = []
    for train_, test_ in kf.split(graph_data_set):
        trainset = []
        test_set = []
        for i in train_:
            trainset.append(graph_data_set[i])
        for j in test_:
            test_set.append(graph_data_set[j])
        training(trainset, test_set, lr, epochs, device, batchsize, num_id, output_list, yy, count)
        num_id = num_id + 1


# draw train and loss graph
def draw(data, color, label):
    ave = np.average(data, axis=0)
    maxx = np.max(data, axis=0)
    minn = np.min(data, axis=0)
    plt.plot(ave, color=color, label=label)
    # plt.plot(maxx, color='r')
    # plt.plot(minn, color='r')
    plt.fill_between(range(60), minn, maxx, facecolor=color, alpha=0.4)
    # plt.show()


if __name__ == '__main__':
    mkdir('results/{}'.format(name))
    mkdir("plots/{}".format(name))
    #train_fun_1(900)

    accurate = []
    recision = []
    ecall = []

    path = "results/{}".format(name)
    roots = ["results/{}/".format(name) + i for i in os.listdir(path)]

    a_train = []
    a_test = []
    l_train = []
    l_test = []

    for i in range(20):
        print("-----------------------{}--------------------------".format(i))
        print("-----------------------{}--------------------------".format(i))
        train_fun_1(i)

        acc_train = []
        acc_test = []
        lo_train = []
        lo_test = []
        for root in roots:
            accuracy_train = torch.load(root + "/acc_train_list{}.pt".format(i))
            loss_train = torch.load(root + "/loss_train_list{}.pt".format(i))
            accuracy_test = torch.load(root + "/acc_test_list{}.pt".format(i))
            loss_test = torch.load(root + "/loss_test_list{}.pt".format(i))

            acc_train.append(accuracy_train[-1])
            acc_test.append(accuracy_test[-1])
            lo_train.append(loss_train[-1])
            lo_test.append(loss_test[-1])

            # acc_train.append(accuracy_train)
            # acc_test.append(accuracy_test)
            # lo_train.append(loss_train)
            # lo_test.append(loss_test)

        a_train.append(np.average(acc_train, axis=0))
        a_test.append(np.average(acc_test, axis=0))
        l_train.append(np.average(lo_train, axis=0))
        l_test.append(np.average(lo_test, axis=0))

        print(a_test)
        print(np.average(a_test))
        print(np.std(a_test))


    #5fold+20times
    plt.figure(figsize=(10,4))
    plt.rcParams['savefig.dpi'] = 500
    plt.subplot(1, 2, 1)
    draw(a_train,'r','train')
    draw(a_test,'orange','test')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    draw(l_train,'g','train')
    draw(l_test,'b','test')
    plt.legend()
    plt.title('Loss')
    plt.savefig("plots/{}/Figure-conbine2.jpg".format(name))
    plt.show()

    for i in range(20):
        print("---------------------{}----------------------------".format(i))
        print("---------------------{}----------------------------".format(i))
        #train_fun_1()
        accurac,precisio,recal = 0,0,0
        for root in roots:
            accuracy_test = torch.load(root + "/acc_test_list{}.pt".format(i))
            precision_test = torch.load(root + "/precision_test_list{}.pt".format(i))
            recall_test = torch.load(root + "/recall_test_list{}.pt".format(i))
            accurac += accuracy_test[-1]
            precisio += precision_test[-1]
            recal += recall_test[-1]

        print("---------------------{}----------------------------".format(i))
        print("---------------------{}----------------------------".format(i))
        print(accurate)
        accurate.append(accurac/5)
        recision.append(precisio/5)
        ecall.append(recal/5)

    print(np.mean(accurate))
    print(np.std(accurate))
    print(np.mean(recision))
    print(np.std(recision))
    print(np.mean(ecall))
    print(np.std(ecall))

