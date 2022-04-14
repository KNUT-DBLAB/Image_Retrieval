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

#features = torch.FloatTensor(features.todense())  # <class 'torch.Tensor'> (1000,100), sparseMatrix -> Tensor Matrix, torch.float32


# print(len(Images))
# print(Images[0].shape) #torch[100,100]
# print(features.shape) #[100,10]
# print(labels.shape)  #[1000]
# print(labels[0])    #type : tensor


idx_train, idx_val, idx_test = ut.splitDataset(34,200,300,len(Images)) #train/val/test로 나눌 imageId를 가진 배열 return
#ut.toDevice(adj, features, labels, idx_train, idx_val, idx_test) # gpu면 gpu로 cpu면 cpu로~

# print(idx_train)
# print(idx_val)
# print(idx_test)


#데이터셋 분리
# x_train, x_val, x_test = Images[idx_train], Images[idx_val], Images[idx_test]

x_train, x_val, x_test = [], [], []
for i in idx_train :
  x_train.append(Images[i])
for i in idx_val :
  x_val.append(Images[i])
for i in idx_test :
  x_test.append(Images[i])

y_train, y_val, y_test = [], [], []
for i in idx_train :
  y_train.append(torch.FloatTensor(encodedLabels[i]))
for i in idx_val :
  y_val.append(encodedLabels[i])
for i in idx_test :
  y_test.append(encodedLabels[i])


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Dataset 상속
class CustomDataset(Dataset):
  def __init__(self):
    self.x_data = x_train
    self.y_data = [[152], [185], [180], [196], [142]]

  # 총 데이터의 개수를 리턴
  def __len__(self):
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx):
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y



labels = torch.Tensor(labels)
#n_labels = labels.max().item() + 1  # 15
n_labels = 15  # 15
n_features = features.shape[1]  # 10
# seed 고정
torch.manual_seed(34)

# model
#GCN(  (gc1): GraphConvolution (100 -> 20)   (gc2): GraphConvolution (20 -> 15) )
model = md.GCN(nfeat=n_features,
            nhid=100,  # hidden = 16
            nclass=n_labels,
            dropout=0.5)  # dropout = 0.5
optimizer = optim.Adam(model.parameters(),
                       lr=0.001, weight_decay=5e-4)

#criterion = nn.CrossEntropyLoss()


epochs = 1000
print_steps = 100
train_loss, train_acc = [], []
val_loss, val_acc = [], []

evaluation = lambda output, target: float(torch.sum(output.eq(target))) / float(target.size()[0])

for epoch in range(45):
    running_loss = 0.0
    acc = 0.
    correct = 0
    outputs = []
    #for i, data in enumerate(train_loader, 0):
    for i in range(len(x_train)):
        inputs, labels = x_train[i], y_train[i]
        #inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        inputs, labels = Variable(inputs), Variable(y_train[i])
        optimizer.zero_grad()
        output = model(features, inputs)
        # print(outputs)
        outputs.append(output)
        print(output.argmax(1))
        loss = F.nll_loss(output.argmax(1), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        prediction = torch.max(outputs.data, 1)[1]  # first column has actual prob.
        correct += prediction.eq(labels.data.view_as(prediction)).cpu().sum()

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.6f acc : %.6f' % (
                epoch + 1, i + 1, running_loss / 2000, 100 * correct / ((i + 1) * 4)))
            running_loss = 0.0

    print('Finished Training')










output = model(features, adj) #[100,15]
print(output.shape)
# test
samples = 10
# torch.randperm : 데이터 셔플
# idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]
idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]
print(torch.randperm(len(idx_test))[:samples])
print(idx_test)
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
# idx_sample 개수만큼의 utput을 가져오는데, argmax로 output 값 15개(각 cluster에 대한 값) 중 가장 큰 값을 뽑고, 그 위치를 받음
# tolist()로 각 feature 별 model의 output 값을 받음
# for 문을 통해 idx2Lble에서 해당하는 label을 반환받음.
print(df)

sys.exit()




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

features = torch.Tensor(features) # tensor(100x10)

#id X freObj = 이미지와 obj간의 관계 = 1000 x 100
adj = torch.FloatTensor(np.load('./data/idFreFeature.npy'))  # tensor(1000,100)


#
# #Data 분포 보고 싶어서 시각화 해봄.
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# data = adj
# #pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)
# df = pd.DataFrame(data, columns=freObjList)
# print(df)
# ax = sns.heatmap(df)
# plt.show()
#
# sys.exit()


testFile = open('./data/cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = (readFile[1:].replace("'", '').replace(' ', '').split(','))
labels = []
for i in range(1000):
    labels.append(int(label[i]))

#features = torch.FloatTensor(features.todense())  # <class 'torch.Tensor'> (1000,100), sparseMatrix -> Tensor Matrix, torch.float32
labels = torch.LongTensor(labels)


idx_train, idx_val, idx_test = ut.splitDataset(34,200,300,len(features)) #train/val/test로 나눌 imageId를 가진 배열 return
ut.toDevice(adj, features, labels, idx_train, idx_val, idx_test) # gpu면 gpu로 cpu면 cpu로~


n_labels = labels.max().item() + 1  # 15
n_features = features.shape[1]  # 10
# seed 고정
torch.manual_seed(34)

# model
#GCN(  (gc1): GraphConvolution (100 -> 20)   (gc2): GraphConvolution (20 -> 15) )
model = md.GCN(nfeat=n_features,
            nhid=100,  # hidden = 16
            nclass=n_labels,
            dropout=0.5)  # dropout = 0.5
optimizer = optim.Adam(model.parameters(),
                       lr=0.001, weight_decay=5e-4)

epochs = 1000
print_steps = 100
train_loss, train_acc = [], []
val_loss, val_acc = [], []


for i in range(epochs):
    tl, ta = ut.train(model,optimizer,features, adj, idx_train,labels)
    train_loss += [tl]
    train_acc += [ta]
    if ((i + 1) % print_steps) == 0 or i == 0:
        tl, ta = ut.evaluate(idx_train, model,features, adj,labels)
        print(features.shape)
        print("adj : ", adj)
        print("adj : ", adj.shape)
        vl, va = ut.evaluate(idx_train, model,features, adj,labels)
        val_loss += [vl]
        val_acc += [va]

        print('Epochs: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Validation Loss: {:.3f}, Validation Acc: {:.3f}'.format(
                i, tl, ta, vl, va))

output = model(features, adj) #[100,15]
print(output.shape)
# test
samples = 10
# torch.randperm : 데이터 셔플
# idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]
idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]
print(torch.randperm(len(idx_test))[:samples])
print(idx_test)
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

