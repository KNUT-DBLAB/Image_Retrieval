import sys
import GCN as md
import torch
import util as ut
import util2 as ut2
from torch.nn.modules.module import Module
import torch.nn as nn
# import pandas as pd
import torch.optim as optim
import pickle
import torch.nn.functional as F
import numpy as np
from gensim.models import FastText
import torch.utils.data as utils
from torch.autograd import Variable

import torch
from torch.utils.data import Dataset, DataLoader

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

class GraphDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, ):
        super(GraphDataset, self).__init__()
        self.x = x_tensor
        self.y = y_tensor
        y_one_hot = torch.zeros((1000, 15))
        y_one_hot.scatter_(1, y_tensor.unsqueeze(1), 1)
        self.attr = y_one_hot
        self.AdjList = []
        self.classList = []

    def __getitem__(self, index):
        # imgAdj = self.AdjList[idx]
        # label = self.classList[idx]
        imgAdj = self.x[index]
        label = self.y[index]
        attr = self.attr[index]
        return imgAdj, label, attr

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    print(len(Images))
    print(len(labels))
    dataset = GraphDataset(Images, labels)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)

    for epoch in range(2):
        print(f"epoch : {epoch}")
        for adj, label, attr in dataloader:
            print("epoch",epoch, "label : ", label, "onehot ,attr : ", attr)
