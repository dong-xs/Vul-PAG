#encoding:utf-8

import pandas as pd

content1=pd.read_csv('predict.csv')

cveid=content1['ID'].tolist()

tokens=content1['tokens']
predict_label=content1['predict_label'].tolist()    #存放预测标签

origin_label=[]                                     #存放原始标签

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

org_data=dataset_get('../generate_data/train_data_zip_spacesplit.txt')
for item in org_data:
    origin_label.append(item[-1])

length=len(origin_label)
diff=[[] for i in range(length)]

for i in range(length):
    pre_lab=eval(predict_label[i])
    org_lab=origin_label[i]
    org_tok=eval(tokens[i])
    diff[i].append(cveid[i])
    for value in range(len(pre_lab)):
        if pre_lab[value]!=org_lab[value]:
            diff[i].append([org_tok[value],(pre_lab[value],org_lab[value])])   #存储每个预测错误的项

with open('org_pred_diff.txt','w') as f:
    for i in diff:
        f.write(str(i)+'\n')

print('写入成功！')