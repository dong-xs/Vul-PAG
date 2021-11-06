import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
import numpy as np
import pandas as pd
import spacy

def dataset_get(filename):
    data = open(filename, 'r',encoding='utf-8')
    content = data.readlines()
    data.close()

    indexes = [0]
    for i in range(len(content)):
        if content[i] == '\n':  # �ҵ�ÿ��λ��Ϊ'\n'������
            indexes.append(i)

    indexes.append(-1)

    for value in range(2, len(indexes) - 1, 2):
        indexes[value] += 1

    sentence_label = []  # �����洢ÿ���־��Ľ��
    for value in range(0, len(indexes) - 1, 2):
        sentence_label.append(content[indexes[value]:indexes[value + 1]])

    sent_length = len(sentence_label)
    # ��������Ҫ�������ÿ���ı�����ת��
    # ����Ϊ���ӳ��ȣ�ÿ��λ���ϵ�Ԫ���������ֹ��ɣ���token���к�label���й���
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


train_data=dataset_get('../generate_data/train_data_zip.txt')
test_data=dataset_get('../generate_data/test_data_zip.txt')

class MyDataset(Dataset):
    def __init__(self):
        self.x_data=train_data
        self.y_data=test_data
        self._len=len(train_data)

    def __getitem__(self, item):
        return self.x_data[item],self.y_data[item]

    def __len__(self):
        return self._len

data=MyDataset()
print(len(data))

first_row=next(iter(data))      #��ȡ��һ�е�����
print(first_row)

dataloader=DataLoader(data,batch_size=3,shuffle=True,drop_last=False,num_workers=1)
for data_val,label_val in dataloader:
    print('x:',data_val,'y:',label_val)
