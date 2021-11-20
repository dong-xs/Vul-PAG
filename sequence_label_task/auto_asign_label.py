# encoding:utf-8

'''
    本程序的目的是：
        当前标注的数据是以description为原数据，然后形成VT,VN,VV等标签的
        那么需要将这些筛选出来的标签转换为BIO的标准格式
'''

import pandas as pd
import spacy

# spacy_nlp = spacy.load('en_core_web_md')
content = pd.read_csv('labeled_data_2200-1.csv', encoding='gb2312')
# content = pd.read_csv('train_data.csv', encoding='gb2312')
# content = pd.read_csv('test_data.csv', encoding='gb2312')

description = content['description']
cveid = content['ID']
v_name = content['VN(name)']
v_version = content['VV(version)']
v_type = content['VT(type)']
v_root_cause = content['VRC(root cause)']
v_position = content['VP(position)']
v_attacker_type = content['VAT(attacker type)']
v_attack_vector = content['VAV(attack vector)']
v_result = content['VR(result)']

import operator
import re

def spilt_sentence(text):
    re_tokens = [t for t in re.split(r'[\s+]', text.replace('\"', '')) if t]  # 判断是否为空时为了处理句尾有空格的情况
    tokens = []
    for token in re_tokens:
        if token[-1] not in [',', '.'] and '(' not in token and ')' not in token:
            tokens.append(token)
            continue
        temp = []
        if token[-1] in [',', '.']:
            temp.append(token[-1])
            token = token[:-1]
        token = token.replace('(', '( ').replace(')', ' )')
        token = [t for t in re.split(r'[\s+]', token) if t]
        tokens.extend(token)
        tokens.extend(temp)
    # print(tokens)
    return tokens

def index_get(sent, v_tag, B_tag, I_tag):
    # 输入为句子、划分好的种类、种类的开始标签、种类的中间标签
    # 输出为特定类别在句子中出现的位置及其对应的标签
    # 已经假定了v_tag是存在的，因此需要先对v_tag是否为进行判断，如果在标注数据中不存在这些标签，则返回None
    print(sent)
    if isinstance(v_tag, str):
        sent_data=spilt_sentence(sent)
        label = ['O' for index in range(len(sent_data))]  # 构造一个全为‘O’的标签
        new_v_tag = v_tag.split(' |n ')  # 存在标注标签交错的情况，尤其针对VN和VV，因此加上了“ \n ”做为分割符

        for v_tags in new_v_tag:
            # tag_doc = spacy_nlp(v_tags.strip())
            # tag_doc=v_tags.strip().split(' ')
            tag_doc=spilt_sentence(v_tags.strip())
            tag_data = [str(tokens) for tokens in tag_doc]

            str_len = len(tag_data)  # 标签字符的长度值
            base_index = sent_data.index(tag_data[0])  # 找到标签首字母出现的首个位置
            while base_index < len(sent_data):
                if operator.eq(sent_data[base_index:base_index + str_len], tag_data) and label[base_index] == 'O':
                    label[base_index] = B_tag  # 起始位置的索引
                    for item in range(base_index + 1, base_index + str_len):
                        label[item] = I_tag  # 终止位置的索引
                    break
                else:
                    base_index += 1
        return label  # 返回标记为v_tag的BIO模式，其余位置设置为O
    else:
        return None


def merge_label(VN_label, VV_label, Vtype_label, Vroot_cause_label, VP_label, Vattacker_type_label,
                Vattack_vector_label, VR_label):
    # 输入为8种标签的8个label序列
    # 输出为一个合成后完整的label序列
    # 先筛选出不为空的候选标签列表
    useful_label = []
    if VN_label is not None:
        useful_label.append(VN_label)
    if VV_label is not None:
        useful_label.append(VV_label)
    if Vtype_label is not None:
        useful_label.append(Vtype_label)
    if Vroot_cause_label is not None:
        useful_label.append(Vroot_cause_label)
    if VP_label is not None:
        useful_label.append(VP_label)
    if Vattacker_type_label is not None:
        useful_label.append(Vattacker_type_label)
    if Vattack_vector_label is not None:
        useful_label.append(Vattack_vector_label)
    if VR_label is not None:
        useful_label.append(VR_label)

    line_length = len(useful_label[0])  # 每个元素标签有多少维
    row_length = len(useful_label)  # 共有多少个元素标签
    label = ['O' for index in range(line_length)]
    for i in range(line_length):
        temp_set = []
        for j in range(row_length):
            temp_set.append(useful_label[j][i])
        for item in temp_set:
            if item != 'O':
                label[i] = item
    return label


# 输出的文件，包括输入和输出两个文件
f_write = open('../generate_data/train_data_zip_spacesplit.txt', 'w')
# f_write = open('../generate_data/test_data_zip.txt', 'w')

for indexes in range(len(description)):
    VN_labels = index_get(description[indexes], v_name[indexes], 'B-VN', 'I-VN')
    VV_labels = index_get(description[indexes], v_version[indexes], 'B-VV', 'I-VV')
    Vtype_labels = index_get(description[indexes], v_type[indexes], 'B-VT', 'I-VT')
    Vroot_cause_labels = index_get(description[indexes], v_root_cause[indexes], 'B-VRC', 'I-VRC')
    VP_labels = index_get(description[indexes], v_position[indexes], 'B-VP', 'I-VP')
    Vattacker_type_labels = index_get(description[indexes], v_attacker_type[indexes], 'B-VAT', 'I-VAT')
    Vattack_vector_labels = index_get(description[indexes], v_attack_vector[indexes], 'B-VAV', 'I-VAV')
    VR_labels = index_get(description[indexes], v_result[indexes], 'B-VR', 'I-VR')

    final_label = merge_label(VN_labels, VV_labels, Vtype_labels, Vroot_cause_labels, VP_labels,
                              Vattacker_type_labels, Vattack_vector_labels, VR_labels)
    # 到这一步已经可以正确的划分出来的，接下来就是输出了，将sent_data和final_label组合起来，然后写入到txt文件中去

    # sent_data = description[indexes].split(' ')
    # sent_data = [str(token) for token in spacy_nlp(description[indexes])]
    sent_data=spilt_sentence(description[indexes])
    for value in range(len(sent_data)):
        temp = '  '.join([sent_data[value], final_label[value], '\n'])
        f_write.write(temp)
    f_write.write('\n')
    f_write.write('\n')
    print('------------------------------------------')
f_write.close()
