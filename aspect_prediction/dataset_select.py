#encoding:utf-8

'''
    本py文件的作用是找出针对各个属性的测试集和训练集
'''
import pandas as pd
from aspect_prediction.attr_asso_rules import *

content=pd.read_csv('../sequence_label_task/labeled_data_2200.csv')
cveid=content['ID']    #所有的CVE ID
description=content['description']     #所有CVE条目的description

def dataset_get(filename):
    data = open(filename, 'r')
    content = data.readlines()
    data.close()

    indexes = [0]
    for i in range(len(content)):
        if content[i] == '\n':  # 找到每个位置为'\n'的索引
            indexes.append(i)

    indexes.append(-1)

    for value in range(2, len(indexes) - 1, 2):
        indexes[value] += 1

    sentence_label = []  # 用来存储每个分句后的结果
    for value in range(0, len(indexes) - 1, 2):
        sentence_label.append(content[indexes[value]:indexes[value + 1]])

    sent_length = len(sentence_label)
    # 接下来需要将里面的每个文本进行转换
    # 长度为句子长度，每个位置上的元素由两部分构成，即token序列和label序列构成
    train_data = [[] for index in range(sent_length)]

    for i in range(sent_length):
        temp_sent = []
        temp_label = []
        for value in sentence_label[i]:
            lists = value.strip().split('  ')
            temp_sent.append(lists[0])
            temp_label.append(lists[1])
        train_data[i] = (temp_sent, temp_label)
    return train_data

data=dataset_get('../generate_data/train_data_zip_spacesplit.txt')
label_list=[]     #存储所有标签
for item in data:
    label_list.append(item[1])   #每个cve条目所对应的标签集

length=len(description)

B_VAT_cveid_label=[]    #存储所有包含B_VAT的条目，存储形式为：cveid-labels
B_VR_cveid_label=[]     #存储所有包含B_VR的条目
B_VAV_cveid_label=[]    #存储所有包含B_VAV的条目

def cveid_label_get(str1,labellists,cveids):
    str_label_cveid=[]
    for i in range(length):
        if str1 in labellists[i]:
            temp_list = []
            for item in labellists[i]:
                if item.startswith('B'):
                    temp_list.append(item)
            str_label_cveid.append((cveids[i], temp_list))
    return str_label_cveid

B_VAT_cveid_label=cveid_label_get('B-VAT',label_list,cveid)
# B_VR_cveid_label=cveid_label_get('B-VR',label_list,cveid)
# B_VAV_cveid_label=cveid_label_get('B-VAV',label_list,cveid)


'''
#基于关联规则筛选各个属性的big rules
B_VAT_data_set = [labels[1] for labels in B_VAT_cveid_label]
B_VR_data_set = [labels[1] for labels in B_VR_cveid_label]
B_VAV_data_set = [labels[1] for labels in B_VAV_cveid_label]

def function(dataset,k_degree,min_support,min_conf,output_item):
    L, support_data = generate_L(dataset, k_degree, min_support)
    big_rules_list = generate_big_rules(L, support_data, min_conf)
    #返回在满足支持度情况下满足最小置信度的所有项集，返回结果包括三个部分：规则项，导出项，置信度评分
    temp_item=[]
    for item in big_rules_list:
        if len(list(item[1]))==1 and list(item[1])[0]==output_item and len(item[0])==k_degree-1:
            temp_item.append(item)

    return temp_item
'''
'''
    规则一：
    设置如下支持度和置信度时：
    print(function(B_VR_data_set,4,0.67,1.0,'B-VR'))
    print(function(B_VAV_data_set,4,0.86,1.0,'B-VAV'))
    print(function(B_VAT_data_set,4,0.75,1.0,'B-VAT'))
    可得到如下规则：
    [(frozenset({'B-VN', 'B-VAT', 'B-VAV'}), frozenset({'B-VR'}), 1.0), (frozenset({'B-VN', 'B-VAT', 'B-VV'}), frozenset({'B-VR'}), 1.0)]
    [(frozenset({'B-VN', 'B-VAT', 'B-VR'}), frozenset({'B-VAV'}), 1.0)]
    [(frozenset({'B-VN', 'B-VR', 'B-VAV'}), frozenset({'B-VAT'}), 1.0), (frozenset({'B-VN', 'B-VR', 'B-VV'}), frozenset({'B-VAT'}), 1.0)]
    针对VR，选择VN、VAT、VAV；
    针对VAV，选择VN、VAT、VR；
    针对VAT，选择VN、VR、VAV；
'''
'''
    规则二：
    设置如下支持度和置信度时：
    print(function(B_VR_data_set,4,0.7,1.0,'B-VR'))
    print(function(B_VAV_data_set,4,0.86,1.0,'B-VAV'))
    print(function(B_VAT_data_set,4,0.8,1.0,'B-VAT'))
    可得到如下规则：
    [(frozenset({'B-VAT', 'B-VN', 'B-VV'}), frozenset({'B-VR'}), 1.0)]
    [(frozenset({'B-VAT', 'B-VN', 'B-VR'}), frozenset({'B-VAV'}), 1.0)]
    [(frozenset({'B-VN', 'B-VV', 'B-VR'}), frozenset({'B-VAT'}), 1.0)]
    针对VR，选择VN、VAT、VV；
    针对VAV，选择VN、VAT、VR；
    针对VAT，选择VN、VR、VV；
'''
