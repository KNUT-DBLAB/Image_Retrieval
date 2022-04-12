import numpy as np
import pandas as pd
import json
from openpyxl import Workbook
import util as ut
from gensim.models import FastText
from tqdm import tqdm
import util as ut
from collections import Counter

np.set_printoptions(linewidth=np.inf)

''' 1000개의 이미지의 빈출 objName '''
def adjColumn(imgCount):
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        object = []
        for i in range(imgCount):
            objects = data[i]["objects"]
            for j in range(len(objects)):  # 이미지의 object 개수만큼 반복
                object.append(objects[j]['names'])
        object = sum(object, [])
        count_items = Counter(object)
        frqHundred= count_items.most_common(n=1000)
        adjColumn = []
        for i in range(len(frqHundred)):
            adjColumn.append(frqHundred[i][0])
        return adjColumn


''' adj 생성(이미지 하나에 대한) '''
def createAdj(imageId, adjColumn):
    with open('./data/scene_graphs.json') as file:  # open json file
        data = json.load(file)
        # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
        # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦

        # imgId의 relationship에 따른 objId, subjId list
        # i는 image id
        imageDescriptions = data[imageId]["relationships"]
        object = []
        subject = []

        for j in range(len(imageDescriptions)):  # 이미지의 object 개수만큼 반복
            object.append(imageDescriptions[j]['object_id'])
            subject.append(imageDescriptions[j]['subject_id'])

    with open('./data/objects.json') as file:  # open json file
        data = json.load(file)
        # 각 이미지 별로 obj, relationship 가져와서 인접 행렬을 만듦
        # 해당 모듈은 이미지 하나에 대한 인접행렬 만듦

        # imgId의 relationship에 따른 objId, subjId list
        # i는 image id
        # objectId = data[imgId][""]
        objects = data[1]["objects"]
        objIdName = []

        for i in range(len(objects)):
            objectsId = objects[i]["object_id"]
            objectsName = objects[i]["names"]
            objIdName.append((objectsId, objectsName))

        # ObjectName은 list 형태임. 여러 개의 이름을 갖는 경우가 있음. 그래서 list에 있는지로 파악해야함
        dictionary = dict(objIdName)
        adjMatrix = np.zeros((len(adjColumn), len(adjColumn)))

        for i in range(len(dictionary)):
            for j in range(len(object)):
                objName = ''
                subName = ''
                if object[j] in dictionary:
                    objName = dictionary[object[j]]
                if subject[j] in dictionary:
                    subName = dictionary[subject[j]]
                if (objName != '') & (subName != ''):
                    if (objName in adjColumn) & (subName in adjColumn):
                        adjI = adjColumn.index(objName)
                        adjJ = adjColumn.index(subName)
                        adjMatrix[adjI][adjJ] += 1
        for i in range(len(adjMatrix[0])) :
            for j in range(len(adjMatrix[1])) :
                if i==j :
                    adjMatrix[i][j] += 1

        return adjMatrix


''' 
feature matrix 2안 
scene graph에서 object-predicate-subject를 scenetence로 묶어서 임베딩 
-> 질문 : 이때 각 단어에 대한 임베딩은 어케 구할건지? 
    일일이 비교해서 구해야 하는지? 
    word가 아니고 phrase인 경우에는? 
    padding?
    '''


