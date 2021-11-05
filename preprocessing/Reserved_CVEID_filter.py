# encoding:utf-8
import pandas as pd

'''
    此代码用于过滤掉那些以“REJECT”和“RESERVED”开头的CVE内容
    输入文件：cve_items.csv
    输出文件：filtered_cve_items.csv
    结果：经过过滤后，删除掉了40000多条数据
'''

data = pd.read_csv('cve_items.csv')  # 读取数据
cve_id = data['id']  # 获取所有id
descriptions = data['description']  # 获取所有的description值
wrong_id = []  # 用于存储包含以“** REJECT **”和“** RESERVED **”开头的CVE条目
data1 = data.copy()  # 将data数据复制一份到data1

for item, cveid in zip(descriptions, cve_id):
    if item[:2] == '**' or item[:3] == '"**' or item[:10] == 'DEPRECATED':
        indexes = data1[data1['id'] == cveid].index
        data1.drop(labels=indexes, inplace=True)
        print(indexes, "处理完成")

print(data1.shape)

data1.to_csv('filtered_cve_item.csv', index=False, header=True)  # 将过滤后的结果写入filtered_cve_item.csv文件
