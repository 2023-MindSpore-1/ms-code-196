# -*- coding: UTF-8 -*-
import sys
import mindspore as ms
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ms.set_context(device_target="GPU")
#此处源于对disa_util_ms的调用，其在bert文件夹下易实现
sys.path.append("../SememeWSD-main/bert/")
from disa_util_ms import *
from tqdm import tqdm

hownet_dict = OpenHowNet.HowNetDict()
check_pos_list = ['noun', 'verb', 'adj', 'adv']
#作为一种全局声明
sememe_model = bert_filter(bert_net_cfg, False)
with open("aux_files/senseid.pkl", 'rb') as f:
    word_sense_id_sem = pickle.load(f)
with open("aux_files/word_candidate.pkl", 'rb') as f:
    word_candidate = pickle.load(f)

# pos_dict = {
#         'n':'noun',
#         'v':'verb',
#         'vn':'verb',
#         'vd':'verb',
#         'vg':'verb',
#         'ns':'noun',
#         'nr':'noun',
#         'a':'adj',
#         'an':'adj',
#         'd':'adv',
#         'ad':'adv',
#     }


correct = 0
count = 0
sense_num = 0
# for item in all_data:

noun_count = 0
noun_correct = 0

verb_count = 0
verb_correct = 0

sememes_dict = {}
sense_dict = {}
# for idx in range(len(all_data)):
#     item = all_data[idx]
#     target_word = item['target_word']
#     sememes_dict[target_word] = hownet_dict.get_sememes_by_word(target_word, structured=False, lang="zh", merge=False)
#     sense_dict[target_word] = []
#     for i in range(len(hownet_dict[target_word])):
#         sense_dict[target_word].append(hownet_dict[target_word][i]['Def'].split(':')[0].replace('{', '').replace('}',''))
i = 0


# model = chinese_bert_wwm_filter()
# sememe_model = chinese_bert_wwm_filter()
def get_disambiguation_word():
    print("===disambiguation===")

    inputf = open("aux_files/simPosSeg.txt", 'r')  # 加载数据集

    outputf = open("aux_files/sim_disambiguation_corpus.txt", 'w', encoding='utf-8')
    wiki_data = inputf.readlines()
    # wiki_data = wiki_data[14981:]
    w = tqdm(wiki_data)
    sentence = []
    for line in w:
        item = eval(line.strip())
        # print(idx,'/',len(all_data))
        context = item['context'][:]
        idx = item['index']
        if idx is -1:
            outputf.write(' '.join(sentence) + '\n')
            sentence = context
            continue
        pos_list = item['part-of-speech']
        target_word = item['target_word']
        sememes_dict[target_word] = hownet_dict.get_sememes_by_word(target_word, structured=False, lang="zh",
                                                                    merge=False)
        target_position = item['target_position']
        target_word_pos = item['target_word_pos']
        # sense_set = item['sense']
        # if '?' in sense_set:
        #     sense_set.remove('?')
        if target_word_pos not in check_pos_list:
            print("pos not in valid list: ", target_word_pos)
            continue
        transformed_pos = target_word_pos

        ch_position = 0
        for word in context:
            if word != '<target>':
                ch_position += len(word)
            else:
                break
        context[target_position] = target_word
        char_sentence = []
        for ch in ''.join(context):
            char_sentence.append(ch)
        # print(char_sentence)
        if target_word not in word_candidate:
            continue
        if transformed_pos not in word_candidate[target_word]:
            continue
        sub_dict = word_candidate[target_word][transformed_pos]
        # print(target_word,transformed_pos)
        # print(ch_position)
        if ch_position > 500:
            continue
        target_index, prob_list = model.select_sense(char_sentence, ch_position, target_word, sub_dict)
        select_sem_set = word_sense_id_sem[target_word][transformed_pos][target_index]

        # real_sense_idx = match_set(word_sense_id_sem[target_word][transformed_pos],sense_set)
        # if real_sense_idx == -1:
        #     continue
        for i in range(len(sememes_dict[target_word])):
            if sememes_dict[target_word][i]['sememes'] == select_sem_set:
                select_No = hownet_dict[target_word][i]['No']
                # select_sys = hownet_dict[target_word][i]['sys']
                break
        select_sem = target_word + '=' + select_No
        if idx == 0:
            outputf.write(' '.join(sentence) + '\n')
            sentence = context
        sentence[target_position] = select_sem

    inputf.close()
    outputf.close()
    # result.add_result(target_word,item['context'],target_position,transformed_pos,select_sem, select_No)


def tran_disambiguation(w):
    sentence = []
    for line in w:
        item = line
        # print(idx,'/',len(all_data))
        context = item['context'][:]
        idx = item['index']
        if idx is -1:
            sentence = context
            continue
        pos_list = item['part-of-speech']
        target_word = item['target_word']
        sememes_dict[target_word] = hownet_dict.get_sememes_by_word(target_word, structured=False, lang="zh",
                                                                    merge=False)
        target_position = item['target_position']
        target_word_pos = item['target_word_pos']
        # sense_set = item['sense']
        # if '?' in sense_set:
        #     sense_set.remove('?')
        if target_word_pos not in check_pos_list:
            print("pos not in valid list: ", target_word_pos)
            continue
        transformed_pos = target_word_pos

        ch_position = 0
        for word in context:
            if word != '<target>':
                ch_position += len(word)
            else:
                break
        context[target_position] = target_word
        char_sentence = []
        for ch in ''.join(context):
            char_sentence.append(ch)
        # print(char_sentence)
        if target_word not in word_candidate:
            continue
        if transformed_pos not in word_candidate[target_word]:
            continue
        sub_dict = word_candidate[target_word][transformed_pos]
        # print(target_word,transformed_pos)
        # print(ch_position)
        if ch_position > 500:
            continue

        target_index, prob_list = model.select_sense(char_sentence, ch_position, target_word, sub_dict)
        select_sem_set = word_sense_id_sem[target_word][transformed_pos][target_index]

        # real_sense_idx = match_set(word_sense_id_sem[target_word][transformed_pos],sense_set)
        # if real_sense_idx == -1:
        #     continue
        for i in range(len(sememes_dict[target_word])):
            if sememes_dict[target_word][i]['sememes'] == select_sem_set:
                select_No = hownet_dict[target_word][i]['No']
                # select_sys = hownet_dict[target_word][i]['sys']
                break
        select_sem = target_word + '=' + select_No
        if idx == 0:
            sentence = context
        sentence[target_position] = select_sem

    lines = ' '.join(sentence)
    return lines


def sememe_wsd(w, model):
    sentence = []
    check_pos_list = ['noun', 'verb', 'adj', 'adv']
    sememes_dict = {}
    sense_dict = {}
    select_sem_string = ''
    sem_word = ''
    sem_words = []
    no_word = ''
    no_words = []
    word_vectors = []
    for line in w:
        item = line

        context = item['context'][:]
        idx = item['index']
        if idx is -1:
            sentence = context
            continue
        pos_list = item['part-of-speech']
        target_word = item['target_word']
        sememes_dict[target_word] = hownet_dict.get_sememes_by_word(target_word, structured=False, lang="zh",
                                                                    merge=False)
        target_position = item['target_position']
        target_word_pos = item['target_word_pos']
        if target_word_pos not in check_pos_list:
            print('词性有误')
            sys.exit(-1)
        transformed_pos = target_word_pos

        ch_position = 0
        for word in context:
            if word != '<target>':
                ch_position += len(word)
            else:
                break
        context[target_position] = target_word
        char_sentence = []
        for ch in ''.join(context):
            char_sentence.append(ch)
        if target_word not in word_candidate:
            print('无此目标词')
            sys.exit(-1)
        if transformed_pos not in word_candidate[target_word]:
            print('目标词无此词性')
            sys.exit(-1)
        sub_dict = word_candidate[target_word][transformed_pos]
        #由于运行过慢，只选择前10个词义
        for i in range(len(sub_dict)):
            try:
                sub_dict[i]=sub_dict[i][:10]
            except:
                pass

        try:
            target_index, prob_list = sememe_model.select_sense(char_sentence, ch_position, target_word, sub_dict)
            select_sem_set = word_sense_id_sem[target_word][transformed_pos][target_index]
        except:
            continue
        for i in range(len(sememes_dict[target_word])):
            if sememes_dict[target_word][i]['sememes'] == select_sem_set:
                select_sense = \
                    hownet_dict[target_word][i]['Def'].split(':')[0].replace('{', '').replace('}', '').split('|')[1]
                select_No = hownet_dict[target_word][i]['No']
                select_syn = hownet_dict[target_word][i]['syn']
                break
        select_sem = target_word + '=' + select_No
        if idx == 0:
            sentence = context
        sentence[target_position] = select_sem
        no_word = target_word + '=' + select_No
        if model.has_index_for(no_word):
            continue
        no_words.append(no_word)
        if len(select_syn) > 5:
            select_syn = select_syn[0:5]
        word_list = []
        for syn in select_syn:
            word_list.append(syn['text'].lower())
        word_list = list(set(word_list))
        #此处完成了词向量的修正
        word_vector = model.get_mean_vector(word_list)
        word_vectors.append(word_vector)
    if no_words != []:
        model.add_vectors(no_words, word_vectors)
    model.fill_norms(force=True)
    return sentence