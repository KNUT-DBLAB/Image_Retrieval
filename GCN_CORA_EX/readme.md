예제 코드 : https://github.com/fahim-sikder/Node-Classification-GCN

# GCN - Node Classification
  
   
    

## DataSet : Cora Dataset
> cora.cites : 논문 간 인용관계를 나타냄 → ndarray(5429,2)  
> cora.content : <논문 id> <논문 feature값> < label >  →  ndarray(2078, 1435)  
  
  
---- 
  

## Data Preprocessing

> **AdjMatrix** : id x id             → tensor(2708, 2708)  
> **FeatureMatrix** : id x feature    → tensor(2708, 1433)  
> **Y** : labels                      → tensor(2708, )   

----

## GCN Model   
  
>  
### **GCN Layer** 

    class GraphConvolution(Module):
> single layer neural network  
module 클래스 상속  

      def __init__(self, in_features, out_features, bias=True):  
        super(GraphConvolution, self).__init__()
        self.in_features = in_features 
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) 
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() 

> 매개 변수 및 초기화 변수를 정의하는데 사용  
전체 클래스의 모든 함수 내에서 사용할 수 있는 초기화 변수를 정의하는데 사용  

> **self.weight** : weight 매개변수 설정. GCN을 정의하는 핵심동작  
→ self.weight.torch.mm의 사용범위는 2차원 행렬. input.shape = [B,N,F]면 사용되지 않음
> - parameter : Layer가 아닌 파라미터 값만 갖고 있음. 모듈의 파라미터 값들을 iterator로 반환함. object type은 torch.Tensor임  
module의 attribute로 할당하면 자동으로 파라메터 리스트에 추가됨   

> **self.reset_parameters()** : 매개변수(가중치) 초기화.   대부분 인자로 weight를 받음.  해당 예제에서는 reset_parameters()를 정의해 self.weight를 초기화 함  
→ Gradient vanishing과 exploding을 막기 위해 사용





    def reset_parameters(self): 
        stdv = 1. / math.sqrt(self.weight.size(1)) #stdb : 0.22360679774997896
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

> 초기화 메소드  
self : GraphConvolution (1433 -> 20)  
위와 같이 따로 정의하지 않고 모듈의 reset_parameters()를 호출해 사용 가능


    def forward(self, input, adj):
        support = torch.mm(input, self.weight) 
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

> **support** : input 데이터(feature) X self.weight    
**output** :  Adj X (feature matrix X weight)  
torch.mm : 행렬 곱  
torch.spmm : sparse matrix 곱  

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ') '


  
>

### **Class GCN**
     class GCN(nn.Module):

>  multiple layer neural network  
**nn.Module class** : pytorch에서 model을 만들 때 상속하는 클래스  
> - 딥러닝을 구성하는 Layer의 base class  
> - Input, Output, Forward, backward 등 학습의 대상이 되는 parameter(tensor) 정의

> nfeat = in_features = 입력되는 데이터의 차원  
nhid = hidden layer = 은닉층의 차원  
nclass = out_features = 출력되는 차원 ≈  출력되는 클래스(label)의 개수  
    
    
각 인자의 type은 tensor지만 Tensor를 직접적으로 입력하면 값이 나오지 않음 → Tensor는 미분의 대상이 되지 않기 때문

    def __init__(self, nfeat, nhid, nclass, dropout): 
        super(GCN, self).__init__() 
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass) 
        self.dropout = dropout 

> nfeat : 1433 ,  nhid : 20,  nclass : 7  
> gc1 : (입력 데이터의 차원, hidden layer의 차원) GraphConvolution(1433->20)  
> gc2 : (hidden layer의 차원, 출력 데이터의 차원(논문의 label 종류 수))  GraphConvolution(20->7)

> Dropout  
> - over fitting을 막기 위해서 사용  
> - 학습 시 레어어 간 연결 중 일부를 랜덤하게 삭제해, 일반화 성능이 높아짐


    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training) 
        x = self.gc2(x, adj) 
        return F.log_softmax(x, dim=1) 



> 순전파 학습  
> input type : tensor  
output : tensor  

> dropout을 통해 overfitting을 막음  
> GCN 레이어에 한 번 더 넣음  
