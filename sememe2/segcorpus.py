from gensim.corpora.wikicorpus import extract_pages,filter_wiki
import jieba.posseg as pseg
import jieba
from tqdm import tqdm
import codecs
#import random
pos_dict = {
    'n':'noun',
    'f':'noun',
    's':'noun',
    't':'noun',
    'PER':'noun',
    'LOC':'noun',
    'ORG':'noun',
    'TIME':'noun',
    'ns':'noun',
    'nr':'noun',
    'nt':'noun',
    'nw':'noun',
    'nz':'noun',
    'v':'verb',
    'vn':'verb',
    'vd':'verb',
    'vg':'verb',
    'a':'adj',
    'an':'adj',
    'ad':'adj',
    'd':'adv',

}
with open("targetwords.txt",'r+',encoding = 'utf-8') as f: # 加载数据集
    targetwords = eval(f.read())   #读取的str转换为字典
f.close()
targetwords_word = list(targetwords.keys())

# 让文本只保留汉字
def fenci(context):
    cut_words = pseg.cut(context)
    return cut_words
def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def format_str(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str + i
    return content_str
def insert_targetwords(sentence, pos_list):
    sentences_mask = []
    position = []
    have_targetwords = False
    for i, word in enumerate(sentence):
        pos = pos_list[i]
        sentence_mask = sentence[:]
        if word in targetwords_word and pos in targetwords[word]:
            have_targetwords = True
            sentence_mask[i] = '<target>'
            sentences_mask.append(sentence_mask)
            position.append(i)
    if have_targetwords is False:
        sentences_mask.append(sentence)
    return sentences_mask, position
def preprocess(context):
    context = format_str(context)
    sentence = fenci(context)

    word_list = []
    pos_list = []
    words_list = []
    all_data = []
    for word, flag in sentence:
        word_list.append(word)
        if flag not in pos_dict:
            pos_list.append(flag)
            continue
        pos_list.append(pos_dict[flag])
    words_list, position = insert_targetwords(word_list, pos_list)
    for i in range(len(words_list)):
        if len(position) == 0:
            data = {'context':words_list[i], 'index': -1}
        else:
            data = {'context':words_list[i],'part-of-speech':pos_list, 'target_word':word_list[position[i]],'target_position':position[i], 'target_word_pos':pos_list[position[i]], 'index': i}
        all_data.append(data)
    sentence = ' '.join(word_list)
    return all_data, sentence

def format_str(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str + i
    return content_str

def segment(context):
    context = format_str(context)
    sentence = jieba.lcut(context)
    sentence = ' '.join(sentence)
    return sentence


def get_seg():
    print('===seging===')
    f = open('aux_files/wiki.txt', 'r')
    wiki = f.readlines()# 34722503行，每5000行需要20-30分钟
    outf = codecs.open('data/wikiSeg.txt', 'w', encoding='utf-8')
    w = tqdm(wiki)
    for l in w:
        sentence = segment(l)
        outf.write(sentence+'\n')
    f.close()
    outf.close()

def get_posseg():
    print('===posseging===')
    f = open('aux_files/sim_corpus.txt', 'r', encoding='utf-8')
    wiki = f.readlines()  # 34722503行，每5000行需要20-30分钟
    # print(l, l+step)
    print(len(wiki))
    posf = codecs.open('aux_files/simPosSeg.txt', 'w', encoding='utf-8')
    w = tqdm(wiki)
    length = 0
    i = 0
    for l in w:
        i += 1
        datas, sentence = preprocess(l)
        if len(sentence) == 0:
            continue
        for data in datas:
            posf.write(str(data) + '\n')
    f.close()
    posf.close()
def segprocess(sentence):
    sentence = jieba.lcut(sentence)
    sentence = ' '.join(sentence)


    # return [w for w in sentence if w not in stopwords]
    return sentence

import pandas as pd
train = pd.read_csv('aux_files/train_lcqmc.txt', sep='\t', header=None)
train = train[0:20000]
train.columns=['q1','q2','label']

f = open('aux_files/sim_corpus.txt', 'w', encoding='utf-8')

for t in train['q1']:
    t = segprocess(t)
    f.write(t)
    f.write('\n')
for t in train['q2']:
    t = segprocess(t)
    f.write(t)
    f.write('\n')
f.close()