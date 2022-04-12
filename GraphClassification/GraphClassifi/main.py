import GCN as md
import torch
import util as ut
from torch.nn.modules.module import Module
import torch.nn as nn
#import pandas as pd


# gpu 사용
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

features,adj, labels = ut.loadData() #데이터 불러옴
idx_train, idx_val, idx_test = ut.splitDataset(34,200,300,len(features)) #train/val/test로 나눌 imageId를 가진 배열 return
ut.toDevice(adj, features, labels, idx_train, idx_val, idx_test) # gpu면 gpu로 cpu면 cpu로~


n_labels = labels.max().item() + 1  # 15
n_features = features.shape[1]  # 100
# seed 고정
torch.manual_seed(34)

# model
#GCN(  (gc1): GraphConvolution (100 -> 20)   (gc2): GraphConvolution (20 -> 15) )
model = md.GCN(nfeat=n_features,
            nhid=20,  # hidden = 16
            nclass=n_labels,
            dropout=0.5)  # dropout = 0.5
optimizer = optim.Adam(model.parameters(),
                       lr=0.001, weight_decay=5e-4)

epochs = 1000
print_steps = 100
train_loss, train_acc = [], []
val_loss, val_acc = [], []


for i in range(epochs):
    tl, ta = ut.train()
    train_loss += [tl]
    train_acc += [ta]

    if ((i + 1) % print_steps) == 0 or i == 0:
        tl, ta = evaluate(idx_train)
        vl, va = evaluate(idx_val)
        val_loss += [vl]
        val_acc += [va]

        print(
            'Epochs: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Validation Loss: {:.3f}, Validation Acc: {:.3f}'.format(
                i, tl, ta, vl, va))

output = model(features, adj)

# test
samples = 10
# torch.randperm : 데이터 셔플
# idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]
idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]
print(idx_sample)

idx2lbl = ['0번 그림', '1번 그림', '2번 그림', '3번 그림', '4번 그림', '5번 그림', '6번 그림', '7번 그림'
    , '8번 그림', '9번 그림', '10번 그림', '11번 그림', '12번 그림', '13번 그림', '14번 그림']
# 실제 라벨 10개, predicate 결과 10개

# numpy.argmax() : 주어진 배열에서 가장 높은 값을 가진 인덱스 반환
# pd.tolist() : 동일 위치(레벨)에 있는 데이터끼리 묶어줌
df = pd.DataFrame({'Real': [idx2lbl[e] for e in labels[idx_sample].tolist()],
                   'Pred': [idx2lbl[e] for e in output[idx_sample].argmax(1).tolist()]})
# [idx2lbl[e] for e in output[idx_sample].argmax(1).tolist()] : 예측한 값이 label의 어디에 속하는지
# Pred
# idx_sample 개수만큼의 output을 가져오는데, argmax로 output 값 15개(각 cluster에 대한 값) 중 가장 큰 값을 뽑고, 그 위치를 받음
# tolist()로 각 feature 별 model의 output 값을 받음
# for 문을 통해 idx2Lble에서 해당하는 label을 반환받음.
print(df)
