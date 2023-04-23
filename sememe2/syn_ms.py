from gensim.models import KeyedVectors
from run_model_ms import *
from segcorpus import *
import sys
from sklearn.metrics import accuracy_score

#设置选取的词向量生成长度
vec_len='200'
model = KeyedVectors.load('aux_files/tencent_ailab_zh_d'+vec_len+'_word2vec.model')
train=pd.read_csv('aux_files/train_lcqmc.txt',sep='\t',header=None)
train = train[0:800]
train.columns=['q1','q2','label']
y_pred = []
min = 10000000
max = 0

#以下为添加多义词消歧模型修正词向量的优化实验
'''
for t in tqdm(range(len(train['q1']))): # 0.719 for 100d 0.740 for 200d
    sentence1 = train['q1'][t]
    datas1, sentence1 = preprocess(sentence1)
    sentence1 = sememe_wsd(datas1, model)
    sentence2 = train['q2'][t]
    datas2, sentence2 = preprocess(sentence2)
    sentence2 = sememe_wsd(datas2, model)
    # print(sentence1, sentence2)
    distance = model.wmdistance(sentence1, sentence2, norm=True)
    # print(distance)
    if distance < min:
        min = distance

    if distance > max:
        max = distance
    # 根据代码，仅可打印出距离
    print('距离：',distance)
    y_pred.append(distance)
  
score=accuracy_score(y_pred ,train['label'].values)
print("WMD_"+vec_len+SWSDS"上下文语义相似度计算准确率为",score)  
'''




#以下则为只进行tencent预训练好的模型完成的lcqmc判断实验
for t in tqdm(range(len(train['q1']))): # 0.679 for 100d 0.704(0.721) for 200d
    sentence1 = train['q1'][t]
    sentence1 = segprocess(sentence1)
    sentence2 = train['q2'][t]
    sentence2 = segprocess(sentence2)
    distance = model.wmdistance(sentence1, sentence2, norm=True)
    if distance < min:
        min = distance

    if distance > max:
        max = distance
    y_pred.append(distance)

for i in range(len(y_pred)):
    diff = max - min
    y_pred[i] = (max-y_pred[i])/diff
    y_pred[i] = int(y_pred[i] >= 0.5)
print(y_pred[0:5])


score=accuracy_score(y_pred ,train['label'].values)
print("WMD_"+vec_len+"上下文语义相似度计算准确率为",score)

