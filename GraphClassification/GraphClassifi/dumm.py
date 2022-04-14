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

# y = torch.zeros((1000, 15))
# y[range(len(labels)), labels] = 1
# 원 핫 인코딩
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
    dataset, sampler=train_sampler, batch_size=5, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=5, drop_last=False)


it = iter(train_dataloader)
batch = next(it)
print(batch)

from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

n_labels = 15  # 15
n_features = features.shape[1]  # 10
model = GCN(n_features, 16, n_labels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(20):
    for batched_graph, labels in train_dataloader:
        #pred = model(batched_graph, batched_graph.ndata['attr'].float())
        #https://github.com/dmlc/dgl/blob/master/python/dgl/data/gindt.py
        #model(batched_graph, 라벨의 one hot encoding값)
        pred = model(batched_graph, batched_graph.ndata['attr'].float())
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


num_correct = 0
num_tests = 0
for batched_graph, labels in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata['attr'].float())
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)

print('Test accuracy:', num_correct / num_tests)