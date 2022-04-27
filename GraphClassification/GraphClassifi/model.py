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


# gpu 사용
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

# GCN Layer
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features   # out_features : 20, in_features : 100, self : unable to get repr for <class'__main__.GraphConvolution'>, bias : True
        #self.out_features = out_features+6
        self.out_features = out_features
        # weight reset
        self.weight = Parameter(torch.empty(in_features, out_features)) # 10x hidden -> 10,15  # out_features : 15, in_features : 10, self : GraphConvolution(100->20), bias : True
        self.weight.to(device)

        if bias:
            self.bias = Parameter(torch.empty(out_features))  #weight = 100,16    self.in_features = 100 self.in_features = 100, out_features=22
            self.bias.to(device)
        else:
            self.register_parameter('bias', None).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        self.weight.to(device)
        self.bias.to(device)
        support = torch.mm(input, self.weight).to(device) #adj = 1,100,100  input = 100,10,  self.weight = 10,15
        output = torch.spmm(adj, support).to(device) #adj = 1,100,100, support = 100, 15

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
        self.conv1 = GraphConvolution(in_feats, h_feats).to(device) #h_feats = 100, in_feats = 10, num_classes = 15
        self.conv2 = GraphConvolution(h_feats, h_feats).to(device)
        self.conv3 = GraphConvolution(h_feats, num_classes).to(device)

    def forward(self, g,in_feat ):#in_feat : 100,10, g = 1, 100,100
        h = self.conv1(in_feat, g).to(device)  #g : tensor(1, 100, 100), in_feat : Tensor(100,10)
        h = F.relu(h) # h = 100, 15
        h = self.conv2(h,torch.empty(100,100).to(device)) # 15x 100    100x 100   -> 15x100
        #h = F.relu(h)
        h = self.conv3(h,torch.ones(10,100).to(device))
       # h = self.conv2(g,h)   
        return  F.log_softmax(h, dim=1)