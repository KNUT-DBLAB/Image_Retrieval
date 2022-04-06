import json
import csv
import util as ut
import os
import numpy as np
import YEmbedding as yed

imgCnt = 1000
adjColumn, xWords = ut.adjColumn_kv(imgCnt)


# textEmbedding하기 위한 이미지별 phrase 모아서 xlsx로 변경
jsonpath = './data/region_descriptions.json'
xlxspath = './data/image_regions.xlsx'
ut.jsontoxml(imgCnt, jsonpath, xlxspath)
embedding_clustering = yed.YEmbedding(xlxspath)


idCluster = embedding_clustering[['image_id', 'cluster', 'distance_from_centroid']]
alist = [10]

for i in range(10):
    df_adj, adjMatrix = ut.createAdj(i, adjColumn)
    alist.append([adjMatrix, idCluster['cluster'][i]])

    #print(i,"번째 AdjMatrix : ")
    #print(adjMatrix)

print(alist[0])
print(alist[3])




'''
imgCnt = 1000
adjColumn, xWords = ut.adjColumn_kv(imgCnt)

i=1
df_adj, adjMatrix = ut.createAdj(i, adjColumn)


print(adjMatrix)
print('type(adjMatrix) : ', type(adjMatrix))
print(adjMatrix.tostring())
print(adjMatrix.tolist())
'''

# f1 = open('./data/imgAdjAll.txt', 'w')
# #for i in range(imgCount):
# for i in range(10):
#     df_adj, adjMatrix = ut.createAdj(i, adjColumn)
#     f1.write(adjMatrix+",")
# f1.close()




# with open('./data/region_graphs.json') as file:  # open json file
#     data = json.load(file)
#     dict = {}
#     # dict 생성(obj_id : obj_name)
#     for i in range(imgCount):
#         imageDescriptions = data[i]["objects"]
#         for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
#             obj_id = imageDescriptions[j]['object_id']
#             if (len(imageDescriptions[j]['names']) != 1):
#                 wholeName = str()
#                 for i in range(len(imageDescriptions[j]['names'])):
#                     wholeName += imageDescriptions[j]['names'][i]
#                 lista = []
#                 lista.append(wholeName.replace(' ', ''))
#                 obj_name = lista
#             else:
#                 obj_name = imageDescriptions[j]['names']
#             dict[obj_id] = obj_name
#
#     keys = sorted(dict)
#     val = []
#
#     for i in keys:
#         if (type(dict[i]) == str):
#             val += (dict[i])
#         val += dict[i]
#
#
# f1 = open('./data/adjColumn_new.txt', 'w')
# for i in keys:
#     f1.write(str(i)+",")
# f1.close()
#
# f2 = open('./data/xWords.txt', 'w')
# for i in val:
#     f2.write(i+",")
# f2.close()