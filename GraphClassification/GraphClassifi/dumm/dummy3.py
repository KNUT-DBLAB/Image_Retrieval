import time

import numpy as np

from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

import argparse
import json
import util as ut
import util2

paser = argparse.ArgumentParser()
args = paser.parse_args("")
args.seed = 123
args.val_size = 0.1
args.test_size = 0.1
args.shuffle = True


np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class GCNDataset(Dataset):
    def __init__(self, list_feature, list_adj, list_logP):
        self.list_feature = list_feature
        self.list_adj = list_adj
        self.list_logP = list_logP

    def __len__(self):
        return len(self.list_feature)

    def __getitem__(self, index):
        return self.list_feature[index], self.list_adj[index], self.list_logP[index]


def partition(list_feature, list_adj, list_logP, args):
    num_total = list_feature.shape[0]
    num_train = int(num_total * (1 - args.test_size - args.val_size))
    num_val = int(num_total * args.val_size)
    num_test = int(num_total * args.test_size)

    feature_train = list_feature[:num_train]
    adj_train = list_adj[:num_train]
    logP_train = list_logP[:num_train]
    feature_val = list_feature[num_train:num_train + num_val]
    adj_val = list_adj[num_train:num_train + num_val]
    logP_val = list_logP[num_train:num_train + num_val]
    feature_test = list_feature[num_total - num_test:]
    adj_test = list_adj[num_total - num_test:]
    logP_test = list_logP[num_total - num_test:]

    train_set = GCNDataset(feature_train, adj_train, logP_train)
    val_set = GCNDataset(feature_val, adj_val, logP_val)
    test_set = GCNDataset(feature_test, adj_test, logP_test)

    partition = {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }

    return partition


testFile = open('../data/freObj.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
freObj = (readFile[1:-1].replace("'", '').replace(' ', '')).split(',')
freObj = freObj[:100]

adjList = []

with open('./data/scene_graphs.json') as file:
    data1 = json.load(file)
with open('./data/objects.json') as file:  # open json file
    data2 = json.load(file)
for i in range(1000):
    adj = ut.createAdj_model2(i, freObj,data1, data2)
    adjList.append(torch.FloatTensor(adj))

features = util2.objNameEmbedding(freObj)

testFile = open('../data/cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = (readFile[1:].replace("'", '').replace(' ', '').split(','))
labels = []
for i in range(1000):
    labels.append(int(label[i]))

features = torch.Tensor(features)
#adjList = torch.Tensor(adjList)  #
adjList = torch.tensor([item.detach().numpy() for item in adjList])
labels = torch.Tensor(labels)
#args = torch.Tensor(args)


dict_partition = partition(features, adjList, labels, args)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_atom, act=None, bn=False):
        super(GCNLayer, self).__init__()
        self.use_bn = bn
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.bn = nn.BatchNorm1d(n_atom)
        self.activation = act

    def forward(self, x, adj):
        out = self.linear(x)
        out = torch.matmul(adj, out)
        if self.use_bn:
            out = self.bn(out)
        if self.activation != None:
            out = self.activation(out)
        return out, adj


class SkipConnection(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(SkipConnection, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        out = in_x + out_x
        return out


class GatedSkipConnection(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GatedSkipConnection, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_coef_in = nn.Linear(out_dim, out_dim)
        self.linear_coef_out = nn.Linear(out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        z = self.gate_coefficient(in_x, out_x)
        out = torch.mul(z, out_x) + torch.mul(1.0 - z, in_x)
        return out

    def gate_coefficient(self, in_x, out_x):
        x1 = self.linear_coef_in(in_x)
        x2 = self.linear_coef_out(out_x)
        return self.sigmoid(x1 + x2)


class GCNBlock(nn.Module):

    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, n_atom, bn=True, sc='gsc'):
        super(GCNBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(GCNLayer(in_dim if i == 0 else hidden_dim,
                                        out_dim if i == n_layer - 1 else hidden_dim,
                                        n_atom,
                                        nn.ReLU() if i != n_layer - 1 else None,
                                        bn))
        self.relu = nn.ReLU()
        if sc == 'gsc':
            self.sc = GatedSkipConnection(in_dim, out_dim)
        elif sc == 'sc':
            self.sc = SkipConnection(in_dim, out_dim)
        elif sc == 'no':
            self.sc = None
        else:
            assert False, "Wrong sc type."

    def forward(self, x, adj):
        residual = x
        for i, layer in enumerate(self.layers):
            out, adj = layer((x if i == 0 else out), adj)
        if self.sc != None:
            out = self.sc(residual, out)
        out = self.relu(out)
        return out, adj


class ReadOut(nn.Module):

    def __init__(self, in_dim, out_dim, act=None):
        super(ReadOut, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(self.in_dim,
                                self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = act

    def forward(self, x):
        out = self.linear(x)
        out = torch.sum(out, 1)
        if self.activation != None:
            out = self.activation(out)
        return out


class Predictor(nn.Module):

    def __init__(self, in_dim, out_dim, act=None):
        super(Predictor, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(self.in_dim,
                                self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = act

    def forward(self, x):
        out = self.linear(x)
        if self.activation != None:
            out = self.activation(out)
        return out


class GCNNet(nn.Module):

    def __init__(self, args):
        super(GCNNet, self).__init__()

        self.blocks = nn.ModuleList()
        for i in range(args.n_block):
            self.blocks.append(GCNBlock(args.n_layer,
                                        args.in_dim if i == 0 else args.hidden_dim,
                                        args.hidden_dim,
                                        args.hidden_dim,
                                        args.n_atom,
                                        args.bn,
                                        args.sc))
        self.readout = ReadOut(args.hidden_dim,
                               args.pred_dim1,
                               act=nn.ReLU())
        self.pred1 = Predictor(args.pred_dim1,
                               args.pred_dim2,
                               act=nn.ReLU())
        self.pred2 = Predictor(args.pred_dim2,
                               args.pred_dim3,
                               act=nn.Tanh())
        self.pred3 = Predictor(args.pred_dim3,
                               args.out_dim)

    def forward(self, x, adj):
        for i, block in enumerate(self.blocks):
            out, adj = block((x if i == 0 else out), adj)
        out = self.readout(out)
        out = self.pred1(out)
        out = self.pred2(out)
        out = self.pred3(out)
        return out







def train(net, partition, optimizer, criterion, args):
    trainloader = torch.utils.data.DataLoader(partition['train'],
                                              batch_size=args.train_batch_size,
                                              shuffle=True, num_workers=2)
    net.train()

    train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad() # [21.01.05 오류 수정] 매 Epoch 마다 .zero_grad()가 실행되는 것을 매 iteration 마다 실행되도록 수정했습니다.

        # get the inputs
        list_feature, list_adj, list_logP = data
        list_feature = list_feature.float()

        # list_adj = list_adj.cuda().float()
        # list_logP = list_logP.cuda().float().view(-1, 1)
        # list_feature = list_feature.cuda().float()

        list_adj = list_adj.float()
        list_logP = list_logP.float().view(-1, 1)
        outputs = net(list_feature, list_adj)

        loss = criterion(outputs, list_logP)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(trainloader)
    return net, train_loss


def validate(net, partition, criterion, args):
    valloader = torch.utils.data.DataLoader(partition['val'],
                                            batch_size=args.test_batch_size,
                                            shuffle=False, num_workers=2)
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for data in valloader:
            list_feature, list_adj, list_logP = data
            # list_feature = list_feature.cuda().float()
            # list_adj = list_adj.cuda().float()
            # list_logP = list_logP.cuda().float().view(-1, 1)
            list_feature = list_feature.float()
            list_adj = list_adj.float()
            list_logP = list_logP.float().view(-1, 1)

            outputs = net(list_feature, list_adj)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

        val_loss = val_loss / len(valloader)
    return val_loss


def test(net, partition, args):
    testloader = torch.utils.data.DataLoader(partition['test'],
                                             batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=2)
    net.eval()
    with torch.no_grad():
        logP_total = list()
        pred_logP_total = list()
        for data in testloader:
            list_feature, list_adj, list_logP = data

            list_feature = list_feature.float()
            list_adj = list_adj.float()
            list_logP = list_logP.float()
            # list_feature = list_feature.cuda().float()
            # list_adj = list_adj.cuda().float()
            # list_logP = list_logP.cuda().float()
            logP_total += list_logP.tolist()
            list_logP = list_logP.view(-1, 1)

            outputs = net(list_feature, list_adj)
            pred_logP_total += outputs.view(-1).tolist()

        mae = mean_absolute_error(logP_total, pred_logP_total)
        std = np.std(np.array(logP_total) - np.array(pred_logP_total))

    return mae, std, logP_total, pred_logP_total


def experiment(partition, args):
    args.epoch = 10
    args.train_batch_size = 100
    args.test_batch_size = 100
    args.l2 = 5e-4
    args.lr = 0.001
    args.n_block=1
    args.n_layer=1
    args.bn=True
    args.in_dim=1000
    args.out_dim=100
    args.hidden_dim=16

    args.optim = 'SGD'

    args.n_atom=1
    args.sc = 'gsc'

    args.pred_dim1 = 1
    args.pred_dim2 = 1
    args.pred_dim3 = 1


    net = GCNNet(args)

    criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('In-valid optimizer choice')

    train_losses = []
    val_losses = []

    for epoch in range(args.epoch):  # loop over the dataset multiple times
        ts = time.time()
        net, train_loss = train(net, partition, optimizer, criterion, args)
        val_loss = validate(net, partition, criterion, args)
        te = time.time()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            'Epoch {}, Acc(train/val): {:2.2f}/{:2.2f}, Loss(train/val) {:2.2f}/{:2.2f}. Took {:2.2f} sec'.format(epoch,
                                                                                                                  train_acc,
                                                                                                                  val_acc,
                                                                                                                  train_loss,
                                                                                                                  val_loss,
                                                                                                                  te - ts))

    mae, std, logP_total, pred_logP_total = test(net, partition, args)

    result = {}
    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['mae'] = mae
    result['std'] = std
    result['logP_total'] = logP_total
    result['pred_logP_total'] = pred_logP_total
    return vars(args), result




# net = GCNNet()
# criterion = nn.MSELoss()
# # model = GCNNet(args)
# # optimizer = optim.Adam(model.parameters(),
# #                        lr=0.001, weight_decay=5e-4)
#
#
# train(net,dict_partition,optimizer,criterion, args)
# validate(net, dict_partition,criterion,args)
# test(net,dict_partition,args)
experiment(dict_partition,args)

