import numpy as np
import pandas as pd
import json
from openpyxl import Workbook
from gensim.models import FastText
from tqdm import tqdm
from collections import Counter
import YEmbedding
from visual_genome import api
import visual_genome_python_driver.visual_genome.local as lc
import time
import torch


np.set_printoptions(linewidth=np.inf)

''' 1000개의 이미지의 빈출 objName '''


def prequency_feature(startId, endId):
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        object = []
        for i in range(endId - startId):
            objects = data[i]["objects"]
            for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
                object.append(objects[j]['names'])
        object = sum(object, [])
        count_items = Counter(object)
        frqHundred = count_items.most_common(100)
        adjColumn = []
        for i in range(len(frqHundred)):
            adjColumn.append(frqHundred[i][0])

        return adjColumn


'''adjMatrix 생성
list = imgId에 대한 cluster 리스트
클러스터링 값이 같으면 += 1로 인접을 표현하고 자기 자신에 대해서도 1값을 추가함
'''


def create_adjMatrix(clusterList):
    adjM = np.zeros((len(clusterList), len(clusterList)))
    for i in range(len(clusterList)):
        for j in range(len(clusterList)):
            if clusterList[i] == clusterList[j]:
                adjM[i][j] += 1
            if i == j:
                adjM[i][j] += 1
    return adjM


''' imageFeature 생성(이미지 하나에 대한) '''


def featuremap(startId, endId, freObjList):
    featureMatrix = []
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        object = []
        for i in range(endId - startId):  # 이미지 1000개에 대한 각각의 objectNamesList 생성
            objects = data[i]["objects"]
            for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
                object.append(objects[j]['names'])  # 이미지 하나에 대한 objList

            # object = sum(object, [])

            # 이미지 하나의 obj랑 최빈 obj 랑 일치하는 게 있으면 1로 표시해서 특징 추출

            row = [0 for i in range(len(freObjList))]
            l = 0
            for k in range(len(freObjList)):
                for j in range(len(objects)):
                    n = ''.join(object[j])
                    m = freObjList[k]

                    if n in freObjList:
                        w = freObjList.index(n)
                        row[w] = 1

            featureMatrix.append((row))
    featureMatrix = np.array(featureMatrix)

    return featureMatrix


''' adj 생성(이미지 하나에 대한) '''


def createAdj(imageId, adjColumn):
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
        # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦

        # i는 image id
        imageDescriptions = data[imageId]["relationships"]
        object = []
        subject = []

        for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
            object.append(imageDescriptions[j]['object_id'])
            subject.append(imageDescriptions[j]['subject_id'])

        adjMatrix = np.zeros((len(adjColumn), len(adjColumn)))
        data_df = pd.DataFrame(adjMatrix)
        data_df.columns = adjColumn
        data_df = data_df.transpose()
        data_df.columns = adjColumn

        # ralationship에 따른
        for q in range(len(object)):
            row = adjColumn.index(object[q])
            column = adjColumn.index(subject[q])
            adjMatrix[column][row] += 1

        return data_df, adjMatrix


''' obj name 단순 임베딩(fasttext로 임베딩 한 값)'''


def objNameEmbedding(xWords):
    a = []
    a.append(xWords)
    # model = FastText(a, vector_size=10, workers=4, sg=1, word_ngrams=1)
    model = FastText(xWords, vector_size=10, workers=4, sg=1, word_ngrams=1)

    # for i in a :
    embedding = []
    for i in xWords:
        embedding.append(model.wv[i])

    return embedding


''' adj 생성(이미지 하나에 대한) '''


def createAdj_model2(imageId, adjColumn, sceneGraph, objJson, ):
    adjM = np.zeros((len(adjColumn), len(adjColumn)))
    # 이미지 내 object, subject 끼리 list 만듦. 한 relationship에 objId, subId 하나씩 있음. Name은 X


    # imgId의 relationship에 따른 objId, subjId list
    # i는 image id
    # imageDescriptions = data[imageId-1]["relationships"]
    imageDescriptions = sceneGraph[imageId - 1]["relationships"]
    objectId = []
    subjectId = []

    for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
        objectId.append(imageDescriptions[j]['object_id'])
        subjectId.append(imageDescriptions[j]['subject_id'])
    # object = sum(object, [])

    # 이미지 하나의 obj랑 최빈 obj 랑 일치하는 게 있으면 1로 표시해서 특징 추출
    # obj에서 각 id로 objName, subName 찾아서 리스트로 저장
    # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
    # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦
    # imgId의 relationship에 따른 objId, subjId list
    # i는 image id
    # objectId = data[imgId][""]

    # 한 이미지 내에서 사용되는 obj의 Id 와 이름 dict;  여러 관계 간 동일 obj가 사용되는 경우가 있기 때문
    # subject의 id값을 넣었을 때 name이 제대로 나오는 지 확인 :
    objects = objJson[imageId - 1]["objects"]
    allObjName = []
    for i in range(len(objects)):
        allObjName.append(([objects[i]['names'][0]], objects[i]['object_id']))
        if not objects[i]['merged_object_ids'] != []:  # id 5090처럼 merged_object_id에 대해서도 추가해주면 좋을 듯
            for i in range(len(objects[i]['merged_object_ids'])):
                allObjName.append(([objects[i]['merged_object_ids'][0]], objects[i]['object_id']))

    objIdName = []
    subIdName = []
    for i in range(len(subjectId)):
        objectName = ''
        subjectName = ''
        for mTuple in allObjName:
            if objectId[i] in mTuple:
                objectName = str(mTuple[0][0])
            if subjectId[i] in mTuple:
                subjectName = str(mTuple[0][0])
            if (objectName != '') & (subjectName != ''):
                objIdName.append(objectName)
                subIdName.append(subjectName)
    # 위에서 얻은 obj,subName List로 adjColumn인 freObj에서 위치를 찾음
    for i in range(len(objIdName)):
        adjObj = ''
        adjSub = ''
        if objIdName[i] in adjColumn:
            adjObj = adjColumn.index(objIdName[i])
            adjM[adjObj][adjObj] += 1
        if subIdName[i] in adjColumn:
            adjSub = adjColumn.index(subIdName[i])
            adjM[adjSub][adjSub] += 1
        if (adjObj != '') & (adjSub != ''):
            adjM[adjObj][adjSub] += 1
    adjM = torch.Tensor(adjM)

    return adjM

