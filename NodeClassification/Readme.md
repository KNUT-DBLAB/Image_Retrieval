GNN Node Classification.py(/GCN0411) \n
dataset : Visual Genome.Scence graph \n
 
Adj : id x id (같은 cluster 값을 갖는 경우 1로 체크함), 1000x1000\n
Feature : id x freOBJ(해당 img에 FreOBJ가 있는 경우 1, 없는 경우 0)\n
freObj : 대상이 되는 이미지 1000개의 Scene graph에서 가장 언급량이 많은 Obj 100개\n
Label : 각 id 당 cluster 번호\n
cluster는 bert-base-nli-mean-tokens를 이용해 15개의 클러스터로 분류했음\n
-> 1000개의 region_graph의 phrase 값을 embedding 함\n
-> 이 부분도 relationship이 잘 나타나지 않은 것 같아 아쉬움\n
relationship을 더 잘 활용할 수 있는 걸 하고 싶음\n
모델의 아웃풋으로 15개의 클래스 중에서 가장 유사한 클러스터링 값을 도출하면, 원래 라벨과 비교해서 학습하고,\n
범위1과 범위 2의 예측 클래스 번호끼리 비교해서 TF를 return 함. 그런데 비교하는 부분은 단순 비교\n
