# GraphClassfication
Visual Genome을 사용한 Graph classfication 모델

> 
**모델 목표 : 이미지 간 유사 여부 판별 모델**
  
> 문제를 단순 graph classfication 으로 축소하여 진행함  
> 모델의 아웃풋으로 15개의 클래스 중에서 가장 유사한 클러스터링 값을 예측하면, 원래 라벨과 비교해서 학습하고,  
범위1과 범위 2의 예측 클래스 번호 간 비교를 통해 TF를 return 함   


> 

## Model 1
- CNN의 Input 형태를 참고해 Input을 입력함, 각 이미지의 freObj 간 연관성을 학습해 해당 이미지의 label을 예측함

### **Input data format**

> **Adj** : img x freObj x freObj, freObj가 언급될 때 1로 체크해 이미지 하나당 특징을 표현하고, 해당 이미지를 
>   - freObj : 대상이 되는 이미지 1000개의 Scene graph에서 가장 언급량이 많은 Obj 100개 
> **Feature** : freObjFeature(freObj를 FastText embedding 한 값) 100x10
> **Label** : 각 id 당 cluster 번호, 총 15개의 cluster 종류가 있음 (1000, 1)    
>  - cluster는 bert-base-nli-mean-tokens를 이용해 15개의 클러스터로 분류했음   
>  - region_graph.json에서 이미지 1000개의 phrase 값을 embedding 해 유사 여부를 판단하고 clustering 과정을 통해 15가지 종류로 이미지를 분류함    
      이를 통해 Label 값을 생성함  
      

----  
> 
## Code 관련 내용
> - 
> -

## **Failure Log**

- 


**개선에 관한 의견**
> - ADJ의 적합성에 대한 의문 -> 각 Obj간의 관계를 나타낸 것으로 보면 납득이 가기도..
