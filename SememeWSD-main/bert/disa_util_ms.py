# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 19:12:54 2022

@author: Administrator
"""
"""
#################pre_train bert example on zh-wiki########################
python run_pretrain.py
"""
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src import  BertNetworkForMask,BertotherFormask




import pickle
import OpenHowNet
import collections
from src.model_utils.config import config as cfg, bert_net_cfg

import numpy as np
from mindspore import Tensor
#在bert主要文件下面，添加使用的文件

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

def convert_token(sentence,orig_word,sub_word_list,position,vocab):
    id_list=[]
    id_list.append('[CLS]')
    for i in range(len(sentence)):
        if position==i:
            for _ in range(len(sub_word_list[0])):
                           id_list.append('[MASK]')
        elif i>position and i<position + len(orig_word):
            continue                       
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
        load_path="load/final.ckpt"
        param_dict = load_checkpoint(load_path)
        load_param_into_net(self.bert_model, param_dict)
        for i in self.bert_model.trainable_params():
            i.requires_grad = False
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
        bert_tokens=convert_token(sentence, orig_word, sub_word_list, position,self.tokenizer)
        # print(sentence)
        # print(bert_tokens)
        id_list=[]
        for token in bert_tokens:
            try:
                id_list.append(self.tokenizer[token])         
            except:
                #针对未知的添加
                id_list.append(self.tokenizer['[UNK]'])
        input_ids=Tensor([id_list])
        #注意起始的cls标记
        masked_lm_positions=[i+position+1 for i in range(len(sub_word_list[0]))]
        masked_lm_positions=Tensor([masked_lm_positions])
        outputs = self.bert_model(input_ids,masked_lm_positions = masked_lm_positions)
        pre_scores = outputs.asnumpy()
        all_probs = []

        char_sub = [list(sub_word) for sub_word in sub_word_list]
        subword_ids=[]
        for x in char_sub:
            temp=[]
            for char in x:
                try:
                    temp.append(self.tokenizer[char])
                except:
                    temp.append(self.tokenizer['[UNK]'])
            subword_ids.append(temp)
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
    def predict_synset_prob_batch(self,sentence,position,orig_word,sub_list):
        new_sen = sentence.copy()
        prob_list = []
        count = 0
        #将target处替换为mask，并未使用
        for idx in range(position,position + len(orig_word)):
            new_sen[idx] = '[MASK]'
        
        sub_length = [len(x) for x in sub_list]
        index_dict = {}
        for idx in range(len(sub_length)):
            length = sub_length[idx]
            if length not in index_dict:
                index_dict[length] = [idx]
            else:
                index_dict[length].append(idx)
        prob_list = []
        #对同义词长度进行汇总，按长度计算处MLM得分
        for length, index_list in index_dict.items():
            curr_subwords = [sub_list[x] for x in index_list]
            curr_probs = self.cal_prob_batch(sentence, orig_word,curr_subwords,position)
            prob_list += curr_probs
        return prob_list

    def select_sense(self,sentence, positions, orig_word,sub_dict):
        avg_prob_list = []
        #按语义不断调用同义词计算MLM得分，返回得出的语义及所有MLM得分
        for i in range(len(sub_dict)):
            subwords = sub_dict[i]
            unique_list = []
            for subword in subwords:
                #不使用原词作为替换词
                if subword == orig_word:
                    continue
                else:
                    unique_list.append(subword)

            final_list = unique_list

            prob_list = self.predict_synset_prob_batch(sentence,positions,orig_word,final_list)
            if len(prob_list) >= 1:
                avg_prob_list.append(np.mean(prob_list))
            else:
                #此处可能出现没有同义词，从而直接丢弃语义的情况
                avg_prob_list.append(-10)
        target_index = np.argmax(avg_prob_list)
        print("selection: %d"%(target_index))
        print()
        return target_index,avg_prob_list

def get_annotation(word):
    hownet_dict = OpenHowNet.HowNetDict()
    sememes = hownet_dict.get_sememes_by_word(word,structured = False,lang = 'zh',merge = True)
    if len(sememes) == 1:
        sememe_list = []
        for sem in sememes:
            sememe_list.append(sem)
        return sememe_list[0]
    else:
        return None




