예제 코드 : https://github.com/fahim-sikder/Node-Classification-GCN

#### GCN - Node Classification

DataSet : Cora Dataset
> - cora.cites : 논문 간 인용관계를 나타냄 -> ndarray(5429,2)
> - cora.content : <논문 id> <논문 feature값> <label> 로 구성됨 -> ndarray(2078, 1435)
  
AdjMatrix : id x id             -> tensor(2708, 2708)  
FeatureMatrix : id x feature    -> tensor(2708, 1433)  
Y : labels                      -> tensor(2708, )  

