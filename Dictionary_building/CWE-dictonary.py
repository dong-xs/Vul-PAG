# encoding:utf-8
'''
    此py文件的作用是从CWE文件中找到各个漏洞的专有名词，以及对漏洞的描述中，最常见的二元组、三元组、四元组短语
    输出：cwe_NER_set.txt
    该文件记录了从CWE最新版文件中抽取出的漏洞类型，并经过人工处理后形成漏洞词典。
'''

import pandas as pd

data = pd.read_csv('cwe_info.csv')
del data['_id']

vul_title = data['title']
vul_descripton = data['description']

vul_NER = []  # 用于存储特殊名词

'''
    首先对标题列进行处理:
        一般来说，在某个标题的最后部分会加上括号，括号内的内容就是对漏洞的别称描述；
        其次，很多漏洞是针对同一类型的，但是是不同的位置，此时主要通过一个冒号来连接，则通过split函数即可区分开
'''
for item in vul_title:
    if item[-1] == ')':  # 若以()结尾，则表明存在别名，则先抽取括号里的漏洞别名
        index_sep = item.rfind("(")
        bracket_content = item[index_sep:]
        vul_NER.append(bracket_content[1:-2] if bracket_content[-1] == "'" else bracket_content)
    else:
        vul_NER.extend(item.split(':'))  # 在冒号连接处切分，再将所有值加入到漏洞名词集中

'''
    第二步对description中的内容进行处理
    目的：找到description中共现频率较高的短语，主要找到出现频率最高的二元、三元以及四元短语
    方法：
        （1）将每个句子按符号进行切分，切分符号主要分为逗号、引号、括号
        （2）再将每个切分后的句子转换成二元短语、三元短语、以及四元短语
        （3）对二元、三元、四元短语进行统计，最终得到结果
        
    经过观察发现：
        每一句description都是对一个漏洞的描述，并无特别常见的的多元组短语；
        从每个句子的成分角度出发，若考虑获取每个句子谓语、宾语【经发现，有意义的词语形式常出现在这两个部分】，则也是对每个句子内容的稀疏抽取。
'''
'''
pos_tag=[',','(',')']
for item in vul_descripton:
    temp_split=item.split()
    print(temp_split)
'''
vul_set = list(set(vul_NER))  # 以集合形式存储cwe中的漏洞名词
with open('cwe_NER_set.txt', 'w') as f:
    for item in vul_set:
        f.write(str(item) + '\n')
f.close()

print('写入成功！')
