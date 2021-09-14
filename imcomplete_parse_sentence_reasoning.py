#encoding:utf-8

'''
    该py程序的目的是，根据上一个程序已经筛选出的无法完整解析的句子，查看这些句子无法正确解析的原因。
    判断思路：
        （1）先定位到每一个无法正确解析的索引点
        （2）查看左右各一个位置的值是什么
        （3）判断这些词组里面的共同项有哪些？
    输入：imcomplete_parse_sentence.csv
    输出：imcomplete_parse_position_spacy.csv
'''

import pandas as pd
import spacy

spacy_nlp = spacy.load('en_core_web_md')

data=pd.read_csv('data/imcomplete_parse_sentence.csv')

ids=data['id']
org_contents=data['content']
exception_index=data['index_value']
parse_result=data['deps_value']

length=len(org_contents)

index2str=[[] for i in range(length)]
index2str_besides=[[] for i in range(length)]

for i in range(length):
    exception_index_temp=eval(exception_index[i])
    parse_result_temp=eval(parse_result[i])
    doc = spacy_nlp(org_contents[i])
    org_token = [token for token in doc]
    for item in exception_index_temp:
        index2str[i].append(org_token[item])
        if item !=0:
            temp_left=org_token[item-1]
        if item != len(org_token) - 1:
            temp_right=org_token[item+1]
        index2str_besides[i].append([temp_left,org_token[item],temp_right])
    print(ids[i],'处理完成')

import csv

headers = ['id', 'content', 'index2str', 'index2str_besides']  # 未完全解析表的表头

f = open('data\imcomplete_parse_position_spacy.csv', 'w', encoding='utf-8')

csv_writer1 = csv.writer(f)
csv_writer1.writerow(headers)

if __name__ == '__main__':
    for idz, item,i2s,i2sb in zip(ids, org_contents,index2str,index2str_besides):
        csv_writer1.writerow([idz, item, i2s, i2sb])
        print('%s 处理完成' % idz)

    f.close()

#当前存在的问题是：没有先进行多句判断，导致出现了很多误解析的存在。那么首先应该进行的多句处理。