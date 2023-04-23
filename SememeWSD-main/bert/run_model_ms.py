from disa_util_ms import *
import os
import mindspore as ms


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ms.set_context(device_target="GPU")
model=bert_filter(bert_net_cfg,False)

all_data = []
with open("data/dataset.txt",'r',encoding = 'utf-8') as f:
    for line in f:
        data = eval(line.strip())
        all_data.append(data)
    
pos_dict = {
        'n':'noun',
        'v':'verb', 
        'vn':'verb',
        'vd':'verb',
        'ns':'noun',
        'a':'adj', 
        'an':'adj',
        'd':'adv', 
        'ad':'adv',
    }
check_pos_list = ['noun', 'verb', 'adj', 'adv']

with open("aux_files/senseid.pkl",'rb') as f:
    word_sense_id_sem = pickle.load(f)
with open("aux_files/word_candidate.pkl",'rb') as f:
    word_candidate = pickle.load(f)


correct = 0
count = 0
sense_num = 0
# for item in all_data:

noun_count = 0
noun_correct = 0

verb_count = 0
verb_correct = 0

logger = log_module()

for idx in range(len(all_data)):
    item = all_data[idx]
    print(idx,'/',len(all_data))
    context = item['context']
    pos_list = item['part-of-speech']
    target_word = item['target_word']
    target_position = item['target_position']
    target_word_pos = item['target_word_pos']
    sense_set = item['sense']
    if '?' in sense_set:
        sense_set.remove('?')
    if target_word_pos not in pos_dict:
        print("pos not in valid list: ",target_word_pos)
        continue
    transformed_pos = pos_dict[target_word_pos]
    

    ch_position = 0
    for word in context:
        if word != '<target>':
            ch_position += len(word)
        else:
            break
        
    #将target填回句子，并计算得到了按字的位置
    context[target_position] = target_word
    char_sentence = []
    for ch in ''.join(context):
        char_sentence.append(ch)
    print(char_sentence)
    if len(char_sentence)>120:
        continue
    if target_word not in word_candidate:
        continue
    if transformed_pos not in word_candidate[target_word]:
        continue
    #此处完成了词性的筛选
    sub_dict = word_candidate[target_word][transformed_pos]
    print(target_word,transformed_pos)
    print(ch_position)
    target_index,prob_list = model.select_sense(char_sentence,ch_position,target_word,sub_dict)
    select_sem_set = word_sense_id_sem[target_word][transformed_pos][target_index]

    real_sense_idx = match_set(word_sense_id_sem[target_word][transformed_pos],sense_set)
    if real_sense_idx == -1:
        continue
    logger.add_word(target_word,transformed_pos,target_index, real_sense_idx)

    
    print(target_index, "sememe set: ", select_sem_set)
    print("real sense: ",sense_set)
    if select_sem_set == sense_set:
        print("!")
        correct += 1
        if transformed_pos == 'noun':
            noun_correct += 1
        elif transformed_pos == 'verb':
            verb_correct += 1
        else:
            print("error: ",transformed_pos)
            pause = input("?")
    
    if transformed_pos =='noun':
        noun_count += 1
    elif transformed_pos == 'verb':
        verb_count += 1

    count += 1
    sense_num += len(sub_dict)

print(correct,count)
print(count,sense_num)

print(noun_correct,noun_count)
print(verb_correct,verb_count)

#格式化输出
print("全部多义词消歧acc为",correct/count)
print("动词多义词消歧acc为",verb_correct/verb_count)
print("名词多义词消歧acc为",noun_correct/noun_count)
   