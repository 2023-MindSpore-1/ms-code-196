# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 19:12:54 2022

@author: Administrator
"""
"""
#################pre_train bert example on zh-wiki########################
python run_pretrain.py
"""
import os
import mindspore as ms
import mindspore.communication.management as D
from mindspore.communication.management import get_rank
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.train_thor import ConvertModelUtils
from mindspore.nn.optim import Lamb, Momentum, AdamWeightDecay, thor
from mindspore import log as logger
from mindspore.common import set_seed
from src import BertNetworkWithLoss, BertNetworkMatchBucket, \
    BertTrainOneStepCell,BertPreTraining, \
    BertTrainOneStepWithLossScaleCell, \
    BertTrainAccumulationAllReduceEachWithLossScaleCell, \
    BertTrainAccumulationAllReducePostWithLossScaleCell, \
    BertTrainOneStepWithLossScaleCellForAdam, \
    BertPretrainEval, BertNetworkForMask,BertotherFormask,\
    AdamWeightDecayForBert, AdamWeightDecayOp
from src.dataset import create_bert_dataset, create_eval_dataset
from src.utils import LossCallBack, BertLearningRate, EvalCallBack, BertMetric
from src.model_utils.config import config as cfg, bert_net_cfg
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num
from mindspore import save_checkpoint



#import tensorflow as tf
import os
import numpy as np
import copy
import pickle
import math
#from sklearn.cluster import MeanShift, estimate_bandwidth 
#import tensorflow_hub as hub
#import pandas as pd
import unicodedata
import collections


import numpy as np
from mindspore import Tensor
#在bert主要文件下面，添加使用的文件

ms.set_context(device_target="GPU")
def match_set(set_list, target_set):
    for i in range(len(set_list)):
        if target_set == set_list[i]:
            return i
    return -1

#tokenizer的实现
def convert_to_unicode(text):
    """
    Convert text into unicode type.
    Args:
        text: input str.

    Returns:
        input str in unicode.
    """
    ret = text
    if isinstance(text, str):
        ret = text
    elif isinstance(text, bytes):
        ret = text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    return ret

def convert_token(sentence,orig_word,sub_word_list,position):
    id_list=[]
    id_list.append('[CLS]')
    for i in range(len(sentence)):
        if position==i:
            for _ in range(len(sub_word_list[0])):
                           id_list.append('[MASK]')
        elif i>position and i<position + len(orig_word):
            continue                       
        else:
            if sentence[i] in ['―','-']:
                id_list.append('[UNK]')
            else:
                id_list.append(sentence[i])
    id_list.append('[SEP]')
    return id_list
def vocab_to_dict_key_token(vocab_file):
    """Loads a vocab file into a dict, key is token."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r",encoding='utf-8') as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


class log_module():
    def __init__(self):
        self.word_dict = {}
    
    def add_word(self,word,pos,select_sense, real_sense):
        if word not in self.word_dict:
            self.word_dict[word] = []
        data = {}
        data['word'] = word
        data['pos'] = pos
        data['select_sense'] = select_sense
        data['real_sense'] = real_sense
        self.word_dict[word].append(data)


class Dictionary():
    def __init__(self):
        self.word_dict = {}
        # count = 0
        with open("sgns.merge.word",'r',encoding = 'utf-8') as f:
            next(f)
            for line in f:
                items = line.split()
                # try:
                word = ''.join(items[:-300])
                vectors = np.array([float(x) for x in items[-300:]])
                # except:
                    # print(word)
                    # print(items[:5])
                self.word_dict[word] = vectors
        keys = list(self.word_dict.keys())[:20]
        # print(self.word_dict.keys())
        for key in keys:
            print(key,self.word_dict[key][:10])

    def __call__(self,word):
        if word in self.word_dict:
            return self.word_dict[word]
        else:
            return None

class bert_filter():
    def __init__(self,bert_net_cfg,is_training):
        self.bert_model = BertNetworkForMask(bert_net_cfg, is_training)
        load_path="../load/bertbase_ascend_v180_cnnews128_official_nlp_loss1.5.ckpt"
        param_dict = load_checkpoint(load_path)
        load_param_into_net(self.bert_model, param_dict)
        self.bert_model.set_train(False)
        self.tokenizer = vocab_to_dict_key_token('vocab.txt')
        self.correct = 0
        self.word_count = 0    
        self.sense_num = 0
        self.rand_count = 0
        self.load_dict()
        
    
    def load_dict(self):
        with open("aux_files/senseid.pkl",'rb') as f:
            self.word_sense_id_sem = pickle.load(f)
        with open("aux_files/word_candidate.pkl",'rb') as f:
            self.word_candidate = pickle.load(f)
    def cal_prob_batch(self,sentence,orig_word,sub_word_list,position):
        #按Bert要求的格式转换,同时适应当前的tokenizer
        bert_tokens=convert_token(sentence, orig_word, sub_word_list, position)
        # print(sentence)
        # print(bert_tokens)
        id_list=[]
        for token in bert_tokens:
            id_list.append(self.tokenizer[token])     
        print(id_list)                     
        input_ids=Tensor([id_list])
        #注意起始的cls标记
        masked_lm_positions=[i+position+1 for i in range(len(sub_word_list[0]))]
        masked_lm_positions=Tensor([masked_lm_positions])
        outputs = self.bert_model(input_ids,masked_lm_positions = masked_lm_positions)
        print(outputs)
        pre_scores = outputs.asnumpy()
        print(pre_scores.shape)
        all_probs = []

        char_sub = [list(sub_word) for sub_word in sub_word_list]
        subword_ids = [[self.tokenizer[char] for char in x]for x in char_sub]
        print(subword_ids)
        #完成从计算出的mask结果，通过同义词的索引
        #返回的结果矩阵，第一个维度对应为句子中字的位置，第二个维度为所有可能的字
        for idx in range(len(subword_ids)):
            temp_list = []
            for pos in range(len(sub_word_list[idx])):
                word_id = subword_ids[idx][pos]
                temp_list.append(pre_scores[pos][word_id])
            all_probs.append(np.mean(temp_list))
        # print(all_probs)
        # pause = input("?")
        return all_probs



import numpy as np
from mindspore import Tensor
#在bert主要文件下面，添加使用的文件
ms.set_context(device_target="GPU")
net=bert_filter(bert_net_cfg,False)
sentence=['如', '何', '连', '通', '分', '散', '的', '果', '农', '与', '大', '市', '场', '，', '活', '跃', '区', '域', '经', '济', '，', '各', '级', '政', '府', '部', '门', '费', '尽', '了', '心', '思', '，', '有', '的', '地', '方', '甚', '至', '连', '给', '干', '部', '分', '配', '销', '售', '任', '务', '的', '招', '儿', '都', '使', '上', '了', '。']


orig_word='使'
sub_word_list=['利用']
position=53
all_probs=net.cal_prob_batch(sentence, orig_word, sub_word_list, position)
print(all_probs)
