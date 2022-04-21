import dgl
import torch
import torch.nn as nn
import sys
import GCN as md
import torch
import util as ut
import util2 as ut2
from torch.nn.modules.module import Module
# import pandas as pd
import torch.optim as optim
import pickle
import torch.nn.functional as F
import numpy as np
from gensim.models import FastText
import torch.utils.data as utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from datasetTest import GraphDataset


from torch.nn.parameter import Parameter
import math




with open("./data/frefre1000.pickle", "rb") as fr:
    data = pickle.load(fr)
Images = data

# freObj(100)의 fastEmbedding 값 100 x 10
testFile = open('./data/freObj.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
freObjList = (readFile[1:-1].replace("'", '')).split(',')
freObjList = freObjList[:100]
model = FastText(freObjList, vector_size=10, workers=4, sg=1, word_ngrams=1)

features = []
for i in freObjList:
    features.append(list(model.wv[i]))
features = torch.FloatTensor(features)  # tensor(100x10)

testFile = open('./data/cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = (readFile[1:].replace("'", '').replace(' ', '').split(','))
labels = []
for i in range(1000):
    labels.append(int(label[i]))

# 원 핫 인코딩
# y = torch.zeros((1000, 15))
# y[range(len(labels)), labels] = 1

labels = torch.LongTensor(labels)
dataset = GraphDataset(Images, labels)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)


from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=1, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=1, drop_last=False)


it = iter(train_dataloader)
batch = next(it)

# 여기부터 -
# GCN Layer
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features   # out_features : 20, in_features : 100, self : unable to get repr for <class'__main__.GraphConvolution'>, bias : True
        #self.out_features = out_features+6
        self.out_features = out_features
        # weight reset
        self.weight = Parameter(torch.empty(in_features, out_features)) # 10x hidden -> 10,15  # out_features : 15, in_features : 10, self : GraphConvolution(100->20), bias : True
        if bias:
             self.bias = Parameter(torch.empty(out_features))   #weight = 100,16    self.in_features = 100 self.in_features = 100, out_features=22
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  #adj = 1,100,100  input = 100,10,  self.weight = 10,15
        output = torch.spmm(adj, support) #adj = 1,100,100, support = 100, 15

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ') '

# - 여기까지
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolution(in_feats, h_feats) #h_feats = 100, in_feats = 10, num_classes = 15
        self.conv2 = GraphConvolution(h_feats, num_classes)

    def forward(self, g,in_feat ):#in_feat : 100,10, g = 1, 100,100
        h = self.conv1(in_feat, g)  #g : tensor(1, 100, 100), in_feat : Tensor(100,10)
        h = F.relu(h) # h = 100, 10
        #h = self.conv2(g,torch.ones(100,10))  #in_features = int(16), out_features = (int)21 이라는데 왜?? 값 이상하게 들어가고 있음. 확인 필요
        h = self.conv2(h,torch.ones(10,100))   #in_features = int(16), out_features = (int)21 이라는데 왜?? 값 이상하게 들어가고 있음. 확인 필요
       # h = self.conv2(g,h)   #in_features = int(16), out_features = (int)21 이라는데 왜?? 값 이상하게 들어가고 있음. 확인 필요
        return  F.log_softmax(h, dim=1)


n_labels = 15  # 15
n_features = features.shape[1]  # 10  #features = Tensor(100,10)
model = GCN(n_features, 15, n_labels)   #n_features = 100, n_labels = 15
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(2000):
    #batched_graph : 1,100,100, labels :    attr : 1,15
    for batched_graph, labels,attr in train_dataloader:
        pred = model(batched_graph.squeeze(), features) #Tensor(1,100,100), features = Tensor(100,10)
        #loss = F.cross_entropy(torch.argmax(pred), attr)
        #splits = torch.stack(list(pred[0].split(1)), dim=0)

        loss = F.nll_loss(pred[0], attr.squeeze().long())
        #loss = F.cross_entropy(splits, torch.FloatTensor(attr.squeeze()))
        #loss = F.cross_entropy(pred[1], attr.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

num_correct = 0
num_tests = 0
for batched_graph, labels,attr in test_dataloader:
    pred = model(batched_graph.squeeze(), features)
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)

print('Test accuracy:', num_correct / num_tests)