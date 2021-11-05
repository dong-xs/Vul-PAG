#encoding:utf-8

'''
    该程序的作用是从原始数据中按年份每一年抽取200条数据来做数据标注，
    标注的数据将用于运行序列标注任务。
'''

import pandas as pd
import random

data=pd.read_csv('../data/filtered_cve_item.csv')
cveid=data['id'].tolist()
content=data['description'].tolist()

new_year=['1999']

for item in [i for i in range(0,21)]:
    if item<10:
        new_year.append('200'+str(item))
    else:
        new_year.append('20'+str(item))

selected_cveid=[]
selected_content=[]

#构建每一个类别
for temp_year in new_year:
    templete='CVE-'+temp_year
    res=[idx for idx in cveid if idx.startswith(templete)]
    cveid_temp=random.sample(res,200)
    selected_cveid.append(cveid_temp)   #随机选择200个样本进行标注
    content_temp = []
    for item in cveid_temp:
        content_temp.append(content[cveid.index(item)])
    selected_content.append(content_temp)


import csv

headers1 = ['id', 'content']  # 解析错误表的表头

f1 = open('../data/selected_labeled_data.csv', 'w', encoding='utf-8')

csv_writer1 = csv.writer(f1)
csv_writer1.writerow(headers1)

if __name__ == '__main__':
    for idx, cont in zip(selected_cveid, selected_content):
        for idx_detail,cont_detail in zip(idx,cont):
            csv_writer1.writerow([idx_detail,cont_detail])
            print('%s 处理完成' % idx)
    f1.close()
