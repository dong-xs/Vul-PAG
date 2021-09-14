# encoding:utf-8
'''
    该PY文件的作用主要有两个：
        （1）找到那些未完全解析的句子（已完成）
            规则：若一个句子不是多句，在进行解析时只会出现一个'ROOT'标签，
                 若解析出两个'ROOT'标签，则该句子肯定在某个地方发生了截断，
                 则需要将该句子标记为未完全解析句子，且记录下切分的索引位置
        （2）找到那些无法正确解析的句子（已完成）
            规则：由于现在的解析全是由动词主导的，若解析出来第一个位置上的'ROOT'
                 词性不为动词，则表示该句子解析错误。（需要使用stanford和spacy两个包来验证）
'''

'''
    VB表示动词基本形式
    VBP表示动词非第三人称单数
    VBZ表示动词第三人称单称
    VBG表示动词进行时/动名词
    VBN表示过去分词，若加上“be”动词，则可以作为谓语动词，即：VB+VBN的形式
    VBD表示过去时态
    还有一种情况，若整个句子中都没有动词的时候，可以直接将第一个元组设置为（‘ROOT’，0，0）
'''

'''
    #画dp树的代码
    from nltk.tree import Tree
    def draw_tree(sentence):
        sent_result = nlp.parse(sentence)
        tree = Tree.fromstring(sent_result)
        tree.draw()
'''

import spacy
from stanfordcorenlp import StanfordCoreNLP
import pandas as pd

spacy_nlp = spacy.load('en_core_web_md')
stanford_nlp = StanfordCoreNLP(r'H:\pycharm_project\Vul-PAG\stanford-corenlp-full-2016-10-31', lang='en')

sent6 = 'Buffer overflow in the changevalue function in libcgi.h for Marcos Luiz Onisto Lib CGI 0.1 allows remote attackers to execute arbitrary code via a long argument.'
sent7 = 'Buffer overflow in the French documentation patch for Gnuplot 3.7 in SuSE Linux before 8.0 allows local users to execute arbitrary code as root via unknown attack vectors.'
# 上述两个例子很明显，第一个句子可以正确解析，能找到allow作为root，第二个句子则错误解析，只找到overflow作为root
# 接下来还需要判断解析错误的句子，这种句子出错的原因不明
# 判断规则：若一个句子首次出现ROOT的那个词，词性不为动词，则该句子必定为错误解析
verb_tags = ['VB', 'VBZ', 'VBP']


def search_wrong_parse_sentence(sentence, verb_tag):
    # 先写个spacy版本的
    # doc = spacy_nlp(sentence)
    # deps = [token.dep_ for token in doc]
    # poss = [token.pos_ for token in doc]
    # root_index = deps.index('ROOT', 0) if 'ROOT' in deps else -1
    # if root_index != -1:
    #     if poss[root_index] != 'VERB':  # 若首次发现root位置上的词性不在动词词性内，则认为该句子解析错误
    #         return deps
    #     else:
    #         return -1

    # 写个stanfordcorenlp版本的
    poss = stanford_nlp.pos_tag(sentence)
    deps = stanford_nlp.dependency_parse(sentence)
    root_index = deps[0][-1] - 1  # 找到ROOT所在位置的索引
    if poss[root_index][-1] not in verb_tag:  # 若首次发现root位置上的词性不在动词词性内，则认为该句子解析错误
        return 1
    else:
        return -1


# 寻找那些无法完全解析的句子，这些句子主要是受特殊符号的影响，需要找到这些未完全解析的句子无法解析位置
# 判断规则：若一个句子中出现了两个ROOT，则表示该句子一定当作了两个分别的句子来进行处理，则需要判定该句子分割的索引位置

sent8 = 'Signed integer overflow in the bttv_read function in the bttv driver (bttv-driver.c) in Linux kernel before 2.4.20 has unknown impact and attack vectors.'


# 上述这个例子对stanford来说，会将其当成两个句子来处理，而对spacy来说，则会当成一个句子来处理

def search_incomplete_parse_sentence(sentence):
    root_index_list = []

    # 先写个spacy版本的
    # doc = spacy_nlp(sentence)
    # deps = [token.dep_ for token in doc]
    # if deps.count('ROOT') > 1:
    #     for items in range(len(deps)):
    #         if deps[items] == 'ROOT':
    #             root_index_list.append(items)
    #     return root_index_list, deps
    # else:
    #     return -1

    # 再写个stanford版本的
    deps = stanford_nlp.dependency_parse(sentence)
    deps_list = [token[0] for token in deps]
    if deps_list.count('ROOT')>1:
        for item in range(len(deps_list)):
            if deps_list[item] == 'ROOT':
                root_index_list.append(item)
        return root_index_list,deps
    else:
        return -1


data = pd.read_csv(r'data\filtered_cve_item.csv')
cveids = data['id']
descriptions = data['description']

import csv

headers1 = ['id', 'content', 'index_value', 'deps_value']  # 未完全解析表的表头
headers2 = ['id', 'content', 'deps_value']  # 解析错误表的表头

f1 = open('data\imcomplete_parse_sentence_stanford.csv', 'w', encoding='utf-8')
f2 = open('data\wrong_parse_sentence_stanford.csv', 'w', encoding='utf-8')

csv_writer1 = csv.writer(f1)
csv_writer1.writerow(headers1)
csv_writer2 = csv.writer(f2)
csv_writer2.writerow(headers2)

if __name__ == '__main__':
    for item, ids in zip(descriptions, cveids):
        # root_index, dep_list
        values = search_incomplete_parse_sentence(item)
        # print(values)
        if search_incomplete_parse_sentence(item) != -1:
            csv_writer1.writerow([ids, item, values[0], values[1]])
        else:
            if search_wrong_parse_sentence(item, verb_tags) != -1:
                csv_writer2.writerow([ids, item, search_wrong_parse_sentence(item, verb_tags)])
        print('%s 处理完成' % ids)

    f1.close()
    f2.close()