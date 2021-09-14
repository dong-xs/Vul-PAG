# encoding:utf-8

from nltk import sent_tokenize
import pandas as pd
import csv
import spacy
from stanfordcorenlp import StanfordCoreNLP

data = pd.read_csv('data/filtered_cve_item.csv')
cveids = data['id']
content = data['description']

spacy_nlp = spacy.load('en_core_web_md')
# stanford_nlp = StanfordCoreNLP(r'H:\pycharm_project\Vul-PAG\stanford-corenlp-full-2016-10-31', lang='en')

plunct = ['.', '!', '?']

# verb_tags = ['VB', 'VBZ', 'VBP']
subj_tags = ['nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'agent', 'expl']   #其实英文句子必须包含的成分是主语和谓语
obj_tags = ['dobj', 'dative', 'attr', 'oprd']


def get_all_sentence(sentence, tag_pos):
    # return sent_tokenize(sent)   #事实证明，直接使用sent_tokenize无法很好的完成句子划分
    start = 0
    index_list = []
    for tag in tag_pos:
        while True:
            index = sentence.find(tag, start)
            if index > -1 and index != len(sentence) - 1:  # 若找到的索引位置不在开头也不在末尾
                start = index + 1
                if sentence[index + 1:].strip()[0].isupper() and sentence[index + 1] == ' ':
                    # 若切分点不是位于最后，且下一个非空位置为大写字母，且根据英文书写规则，其写完标点符号后会留一个空格
                    index_list.append(index)
                elif index <= len(sentence) - 2 and sentence[index + 1] == '"':
                    # 还有一种特殊情况就是，一句话的结束也可以用"表示，那么就需要将该"的索引位置记录下来
                    index_list.append(index + 1)
                else:
                    continue
            else:
                break
        # 至今已经找到了所有的index，接下来需要判断两两字符间是否可以构成一个完整的句子

    for index_list

    return index_list


def judge_complete_simple_sentence(sent):
    docs = spacy_nlp(sent)   #构建doc对象
    deps = [item.dep_ for item in docs]   #存放每个词的依赖关系标签
    pos = [item.pos_ for item in docs]    #存放每个词的词性标签
    heads = [item.head.dep_ for item in docs]  # 存放每个词的前一个词的依赖情况

    verb_index = deps.index('ROOT') if pos[deps.index('ROOT')] == 'VERB' else -1    #若某个词的依赖关系为'ROOT'
    # 已定位到动词的索引位置，需要找到所有与该ROOT直接关联的词项
    if verb_index != -1:    #若存在动词
        temp_sub_idx=[]    #存放主语相关的索引位置
        # temp_obj_idx = []  #存放宾语相关的索引位置
        for value in range(len(heads)):
            if heads[value]=='ROOT':
                if deps[value] in subj_tags and value<verb_index:    #若该位置上的词在主语标签内且索引值小于动词的位置
                    temp_sub_idx.append(value)          #存放与ROOT相关主语依赖关系的索引位置
                # elif heads[value] in obj_tags and value>verb_index:  #若该位置上的词在主语标签内且索引值大于动词的位置
                #     temp_obj_idx.append(value)          #存放与ROOT相关宾语依赖关系的索引位置
        if len(temp_sub_idx)!=0:    #主语标签必定不为0，则认为该句子是完整的
            return 1
    else:
        return 0


# def get_imcomplete_sentence(sent,pos_tag):
#     tags=stanford_nlp.pos_tag(sent)    #获取句子的所有词性标注结果
#     #tags=[token.pos_ for token in doc]
#     ret=[i for i in pos_tag if i not in tags]    #求两个列表的交集，
#     return 1 if len(ret)==0 else 0    #若两列表交集的长度为0，则表示没有

# if __name__ == "__main__":
#     headers1 = ['id', 'content', 'split_list']  # 多句话的表头
#     f1 = open('data\multi_sentence_new.csv', 'w', encoding='utf-8')
#     csv_writer1 = csv.writer(f1)
#     csv_writer1.writerow(headers1)
#
#     headers2 = ['id', 'content']  # 完整句子结构的表头
#     f2 = open('data\single_sentence_new.csv', 'w', encoding='utf-8')
#     csv_writer2 = csv.writer(f2)
#     csv_writer2.writerow(headers2)
#
#     for item, ids in zip(content, cveids):
#         if item[0] == '"' and item[-1] == '"':  # 有大部分的description是以引号开头，以引号结尾的，则将这些部分去除掉。
#             item = item[1:-1]
#         # 当前针对每一个description进行多句判断时，会有a.k.a以及e.g.等特定短语干扰判断结果
#         item = item.replace('a.k.a.', 'aka')  # 使用a.k.a.代替aka可以减少按句号切分
#         item = item.replace('e.g.', 'for example')
#         item = item.replace('.. (dot dot)', 'dot dot')
#         item = item.replace('. (dot)', 'dot')
#         item = item.replace('etc .', 'and so on')
#         item = item.replace('etc.', 'and so on')
#         item = item.replace('i.e.', '')
#
#         indexes = get_all_sentence(item)
#
#         if len(indexes) == 1:  # 如果返回的长度为0，则表示检测出来的句子长度为1，必定为一个单句
#             csv_writer2.writerow([ids, item])
#         else:
#             csv_writer1.writerow([ids, item, indexes])
#         print('%s 处理完成' % ids)
