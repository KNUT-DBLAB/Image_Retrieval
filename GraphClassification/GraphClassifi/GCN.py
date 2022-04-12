from torch.nn.modules.module import Module
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import math
import torch.optim as optim
import torch.nn.functional as F





# Model
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features   # out_features : 20, in_features : 100, self : unable to get repr for <class'__main__.GraphConvolution'>, bias : True
        self.out_features = out_features

        # weight reset
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # # out_features : 20, in_features : 100, self : GraphConvolution(100->20), bias : True
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:

            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ') '


class GCN(nn.Module):   #nhid : 20, nfeat : 100, self : GCN(), nclass : 15, dropout : 0.5
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        # self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc1 = GraphConvolution(nfeat, nclass)   #nhid : 20, nfeat : 100, self : GCN(), nclass : 15, dropout : 0.5
                                                    # out_features : 20, in_features : 100, self : unable to get repr for <class'__main__.GraphConvolution'>, bias : True
        self.gc2 = GraphConvolution(nclass, nfeat)

        #nhid : 20, nfeat : 100, self : (gc1) : GraphConvolution(100->20), nclass : 15, dropout : 0.5
                                                     # in_features : 20, out_features : 15,  self : unable to get repr for <class'__main__.GraphConvolution'>, bias : True
        self.dropout = dropout

    # X : 초기 랜덤값 -> 학습 하면서 변경
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)
