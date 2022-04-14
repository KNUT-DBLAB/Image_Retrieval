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



with open("./data/frefre1000.pickle", "rb") as fr:
    data = pickle.load(fr)
Images = data

# gpu 사용
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#freObj(100)의 fastEmbedding 값 100 x 10
testFile = open('./data/freObj.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
freObjList = (readFile[1:-1].replace("'",'')).split(',')
freObjList = freObjList[:100]
a = []
a.append(freObjList)
model = FastText(freObjList, vector_size=10, workers=4, sg=1, word_ngrams=1)

features = []
for i in freObjList:
    features.append(list(model.wv[i]))
features = torch.FloatTensor(features) # tensor(100x10)

testFile = open('./data/cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = (readFile[1:].replace("'", '').replace(' ', '').split(','))
labels = []
for i in range(1000):
    labels.append(int(label[i]))


y = torch.zeros((1000,15))
y[range(len(labels)), labels] = 1
encodedLabels = y


# dataset 구성요소

class my_dataset(torch.utils.data.Dataset):

    def __init__(self, x, transforms=None):

    def __len__(self):

    def __getitem__(self, idx):
