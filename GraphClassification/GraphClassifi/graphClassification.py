from cProfile import label
from tkinter import Image
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
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import model as md
import random

# gpu 사용
USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

features, adj, labels = ut.loadData()

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
features = torch.FloatTensor(features).to(device)  # tensor(100x10)


features, adj, labels  = features.to(device), adj.to(device), labels.to(device)

# testFile = open('./data/cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
# readFile = testFile.readline()
# label = (readFile[1:].replace("'", '').replace(' ', '').split(','))
# labels = []  
# for i in range(1000):
#     labels.append(int(label[i]))

# y = torch.zeros((1000, 15))
# y[range(len(labels)), labels] = 1
# 원 핫 인코딩
# labels = torch.LongTensor(labels).to(device)

with open("./data/frefre1000.pickle", "rb") as fr:
    data = pickle.load(fr)
Images = data

dataset = GraphDataset(Images, labels)

dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train).to(device))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples).to(device))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=1, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=1, drop_last=False)

it = iter(train_dataloader)
batch = next(it)

n_labels = 15  # 15
n_features = features.shape[1]  # 10  #features = Tensor(100,10)

model = md.GCN(n_features, 15, n_labels)  #n_features = 100, n_labels = 15
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#labels, features, model = labels.to(device), features.to(device), model.to(device)
 
for epoch in range(20):
    #batched_graph : 1,100,100, labels :    attr : 1,15
    for batched_graph, labels,attr in train_dataloader:
        batched_graph, labels, attr = batched_graph.to(device), labels.to(device), attr.to(device)
        batched_graph = batched_graph.squeeze().to(device)

        pred = model(batched_graph, features) #Tensor(1,100,100), features = Tensor(100,10)

        #loss = F.nll_loss(pred[0], attr.squeeze().long())
        loss = F.nll_loss(pred[0], attr.squeeze().long()).to(device) 

        print("pred.max : ", pred[0], "labels : ", labels, "attr : ", attr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

num_correct = 0
num_tests = 0

for batched_graph, labels,attr in test_dataloader:
    batched_graph, labels, attr = batched_graph.to(device), labels.to(device), attr.to(device)
    batched_graph = batched_graph.squeeze().to(device)
    attr = attr.squeeze().long().to(device)
    pred = model(batched_graph, features).to(device)
    #num_correct += (torch.argmax(pred[1])== labels).sum().item()
    #num_correct += (pred.argmax(1) == labels).sum().item()
    num_correct += (pred[0] == attr).sum().item()
    print("pred.max : ", pred[0], "labels : ", labels, "attr : ", attr)
    num_tests += len(labels)

print('Test accuracy:', num_correct / num_tests)


##commit 용 dumm

#conv 2
#20 - 0.15
#200 - 0.2
#2000 - 0.25

#conv 3 - ones - relu 없을 때
#20 - 0.2
#200 - 0.2


#conv 3 - empty - relu 없을 때
#20 - 0.55
#200 - 0.7 - 0.3 - 0.7 - 0.1 - 0.25

#conv 3 - empty - 2-3 사이 relu 추가 ~_~
#20 - 0.2
#200 - 0.2

#conv 3 - empty - 1-2 사이 relu 추가 / 1-2 사이 relu  삭제 ~_~
#20 - 0.3 -
#200 - 0.9 - 0.5 - 0.1 - 0.1
