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

# cnn Image Classification - https://ariz1623.tistory.com/302

'''
Model 2
- Adj : id x id
- Feature : img x (freObj x freObj) // 1000x(100x100)
- Lable : 1000 x 1

기존 GCN 코드를 이용해 Scene graph의 realationship이 반영된 GCN 모델 만들기
-> 전처리 말고 직접적으로 모델을 여러 번 학습 시킬 순 없는지? 
지금 사용하는 Adj는 cluster 값을 알아야만 가능한데 각 이미지를 학습시켜서는 안되는지?

'''

with open("./data/graphDataset.pickle", "rb") as fr:
    data = pickle.load(fr)
dataset = data

# gpu 사용
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Adj
## Load pickle -> Image의 freObj x freObj 로 만든 AdjacencyMatrix 1000개 List
with open("./data/frefre1000.pickle", "rb") as fr:
    data = pickle.load(fr)
ImageAdjs = data

# Features
testFile = open('./data/freObj.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
freObj = (readFile[1:-1].replace("'", '')).split(',')
features = ut2.objNameEmbedding(freObj[:100])
features = torch.FloatTensor(features)

# Y - labels, (1000, )
testFile = open('./data/cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = (readFile[1:].replace("'", '').replace(' ', '').split(','))
labels = []
for i in range(1000):
    labels.append(int(label[i]))

labels = torch.LongTensor(labels)

idx_train, idx_val, idx_test = ut.splitDataset(34, 200, 300, len(ImageAdjs))  # train/val/test로 나눌 imageId를 가진 배열 return
# ut.toDevice(adj, features, labels, idx_train, idx_val, idx_test) # gpu면 gpu로 cpu면 cpu로~

n_labels = labels.max().item() + 1  # 15
n_features = features.shape[1]  # 10

# seed 고정
torch.manual_seed(34)

# model
# GCN(  (gc1): GraphConvolution (100 -> 20)   (gc2): GraphConvolution (20 -> 15) )
model = md.GCN(nfeat=n_features,
               nhid=20,  # hidden = 16
               nclass=n_labels,
               dropout=0.5)  # dropout = 0.5

optimizer = optim.Adam(model.parameters(),
                       lr=0.001, weight_decay=5e-4)

epochs = 1000
# print_steps = 100
train_loss, train_acc = [], []
val_loss, val_acc = [], []

loss_ = []  # loss 저장용 리스트
n = len(ImageAdjs)  # 배치개수


for epoch in range(10):  # 10회 반복
    running_loss = 0.0
    outputs = []
    target = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    target = torch.Tensor(target)


    for i, data in enumerate(ImageAdjs, 0):
        # inputs, labels = data[0].to(device), data[1].to(device)  # 배치 데이터
        optimizer.zero_grad()  # 배치마다 optimizer 초기화
        #data = fre x fre Adj
        output = model(features, data)  # 노드 10개짜리 예측값 산출   features : tensor(100,10), data =tensor(100,100)
        outputs.append(int(output.argmax(1)[1]))

        loss = F.nll_loss(output, target)  # 크로스 엔트로피 손실함수 계산

        #loss = F.nll_loss(output, labels)  # 크로스 엔트로피 손실함수 계산
        optimizer.zero_grad()
        loss.backward()  # 손실함수 기준 역전파
        optimizer.step()  # 가중치 최적화
        running_loss += loss.item()

    loss_.append(running_loss / n)
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(ImageAdjs)))

#
# val_loss = []
# val_acc = []
#
# # for j in range(len(ImageAdjs)):
# #     model.train()
# #     adj = ImageAdjs[j]
# #     label = labels[j]
# #     optimizer.zero_grad()
# #     outputs = model(features, adj)
# #     predictLabel = outputs.argmax(1)[0]
# #     loss = F.nll_loss(predictLabel, label)
# #     loss.backward()
# #     optimizer.step()
# #
# #     if (i + 1) % 20 == 0:
# #         print(f'Epoch: {epoch} - Loss: {loss:.6f}')
# #
# #     val_loss = []
# #     val_acc = []
#
#
# # 모델 검증
# for i, (images, targets) in enumerate(valid_loader):
#     model.eval()
#     images, targets = images.to('cuda'), targets.to('cuda')
#
#     with torch.no_grad():
#         outputs = model(images)
#         valid_loss = criterion(outputs, targets).cpu().detach().numpy()
#
#         preds = torch.argmax(outputs, axis=1)
#         preds = preds.cpu().detach().numpy()
#
#         targets = targets.cpu().detach().numpy()
#         batch_acc = (preds == targets).mean()
#
#         val_loss.append(valid_loss)
#         val_acc.append(batch_acc)
#
# val_loss = np.mean(val_loss)
# val_acc = np.mean(val_acc)
#
# print(f'Epoch: {epoch} - valid Loss: {val_loss:.6f} - valid_acc : {val_acc:.6f}')
#
# if valid_loss_min > val_loss:
#     valid_loss_min = val_loss
#     best_models.append(model)
#
# # Learning rate 조절
# lr_scheduler.step()

'''
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
'''
