# encoding:utf-8

from stanfordcorenlp import StanfordCoreNLP
import pandas as pd

'''
    该py文件的作用是：判断每个句子的类型，并分别写入到各个文件中，其判断逻辑为：
        先判断当前句子是否为多句；
        再判断该句子是否为不完整的形式；
        若都不满足上述的两种句子结构，则该句子为正常完整结构的句子。
'''

data = pd.read_csv('data/filtered_cve_item.csv')
cveid = data['id']
content = data['description']

sep_tag = ['.', '!', '?']


# sent1 = 'The GREE application before 1.4.0, GREE Tanken Dorirando application before 1.0.7, GREE Tsurisuta application before 1.5.0, GREE Monpura application before 1.1.1, GREE Kaizokuoukoku Columbus application before 1.3.5, GREE haconiwa application before 1.1.0, GREE Seisen Cerberus application before 1.1.0, and KDDI&GREE GREE Market application before 2.1.2 for Android do not properly implement the WebView class, which allows remote attackers to obtain sensitive information via a crafted application.'
# sent2 = 'Unspecified vulnerability in Oracle Java SE 7 Update 11 (JRE 1.7.0_11-b21) allows user-assisted remote attackers to bypass the Java security sandbox via unspecified vectors, aka "Issue 51," a different vulnerability than CVE-2013-0431.  NOTE: as of 20130130, this vulnerability does not contain any independently-verifiable details, and there is no vendor acknowledgement. A CVE identifier is being assigned because this vulnerability has received significant public attention, and the original researcher has an established history of releasing vulnerability reports that have been fixed by vendors.  NOTE: this issue also exists in SE 6, but it cannot be exploited without a separate vulnerability.'
# sent3 = 'Unspecified vulnerability in Haakon Nilsen simple, integrated publishing system (SIPS) before 0.2.4 has an unknown impact and attack vectors, related to a "grave security fault." This will cause XSS.'
# sent4 = 'Denial of service in Qmail through long SMTP commands.'    #该句子没有动词
# sent5 = 'File creation and deletion, and remote execution, in the BSD line printer daemon (lpd).'     #该句子也没有动词

# sent1为简单句，sent2为多句，sent3为"结尾的单句，以这三个句子为测试句子
# 该函数用于判断某个cveid到底有几句话，并返回每句话切分点的index值。
# def judge_multi_sentence(sentence, tag_pos):
#     start = 0
#     index_list = []
#     for tag in tag_pos:
#         while True:
#             index = sentence.find(tag, start)
#             if index > -1:
#                 start = index + 1
#                 if index != len(sentence) - 1 and sentence[index + 1:].strip()[0].isupper() and sentence[
#                     index - 1] != ' ':  # 若切分点不是位于最后，且下一个非空位置为大写字母
#                     index_list.append(index)
#                 elif index <= len(sentence) - 2 and sentence[
#                     index + 1] == '"':  # 还有一种特殊情况就是，一句话的结束也可以用"表示，那么就需要将该"的索引位置记录下来
#                     index_list.append(index + 1)
#                 else:
#                     continue
#             else:
#                 break
#     return index_list

'''
    #根据EXTRACTOR中关于句子的分解，若一个字符串完整包含主语、谓语、宾语的结构，则认为
'''
def judge_multi_sentence(sentence, tag_pos):
    start = 0
    index_list = []
    for tag in tag_pos:
        while True:
            index = sentence.find(tag, start)
            if index > -1 and index != len(sentence) - 1:
                start = index + 1
                if sentence[index + 1:].strip()[0].isupper() and sentence[index + 1] == ' ':  # 若切分点不是位于最后，且下一个非空位置为大写字母
                    index_list.append(index)
                elif index <= len(sentence) - 2 and sentence[
                    index + 1] == '"':  # 还有一种特殊情况就是，一句话的结束也可以用"表示，那么就需要将该"的索引位置记录下来
                    index_list.append(index + 1)
                else:
                    continue
            else:
                break
    return index_list

# 接下来判断某一个句子是否为复杂句子，判断规则为：若一个description中有多个单数或复数形式的动词，则表示该句子为复杂句子。返回为当前判定为动词的索引值
verb_tags = ['VB', 'VBP', 'VBZ']
stanford_nlp = StanfordCoreNLP(r'stanford-corenlp-full-2016-10-31', lang='en')


def judge_complex_sent(sentence, verb_tag):
    pos_list = stanford_nlp.pos_tag(sentence)  # 此处使用stanford的包是因为，该包可以细化每一个动词的词性到VBP、VBZ等，而spacy只能细化到VERB
    postags = [tag[1] for tag in pos_list]
    counts = []
    for tag in verb_tag:
        counts.append(postags.index(tag))  # 在此处没有考虑情态助动词do、can等作为动词的情况，当前可以忽略该情况
    return counts


# 接下来判断某一个句子的结构是否完整，主要针对当前句子中不存在动词的情况
new_verb_tag = ['VBP', 'VBZ','VB']  # 此处的动词标签去掉VB，是因为VB常会带有助动词，而这些词无法十分明确的表示谓语动词，因此可以删掉


def judge_imcomplete_sentence(sentence, verb_tag):
    pos_list = stanford_nlp.pos_tag(sentence)
    postags = [tag[1] for tag in pos_list]
    count = 0
    for verb in verb_tag:
        if verb in postags:
            count += 1
        else:
            continue
    return count


import csv

headers1 = ['id', 'content', 'index', 'split_list']  # 多句话的表头
f1 = open('data\multi_sentence.csv', 'w', encoding='utf-8')
csv_writer1 = csv.writer(f1)
csv_writer1.writerow(headers1)

headers2 = ['id', 'content']  # 完整句子结构的表头
f2 = open('data\complete_struture_sentence.csv', 'w', encoding='utf-8')
csv_writer2 = csv.writer(f2)
csv_writer2.writerow(headers2)

headers3 = ['id', 'content']  # 不完整句子结构的表头
f3 = open('data\imcomplete_struture_sentence.csv', 'w', encoding='utf-8')
csv_writer3 = csv.writer(f3)
csv_writer3.writerow(headers3)

from itertools import tee  # 该第三方库可以用于对列表元素进行迭代

for item, ids in zip(content, cveid):
    if item[0] == '"' and item[-1] == '"':  # 有大部分的description是以引号开头，以引号结尾的，则将这些部分去除掉。
        item = item[1:-1]
    # 当前针对每一个description进行多句判断时，会有a.k.a以及e.g.等特定短语干扰判断结果
    item = item.replace('a.k.a.', 'aka')  # 使用a.k.a.代替aka可以减少按句号切分
    item = item.replace('e.g.', 'for example')
    item = item.replace('.. (dot dot)', 'dot dot')
    item = item.replace('. (dot)', 'dot')

    indexes = judge_multi_sentence(item, sep_tag)
    if len(indexes) == 0:  # 如果返回的长度为0，则表示检测出来的句子长度为1，必定为一个单句
        if judge_imcomplete_sentence(item, new_verb_tag) == 0:  # 如果返回的是0，则表示句子中没有动词，则将其写入文件3中
            csv_writer3.writerow([ids, item])
        else:  # 如果返回的是其他，则表示句子中存在动词，则将其写入完整结构的文件2中
            csv_writer2.writerow([ids, item])
    elif len(indexes) == 1:  # 如果返回的长度为1，则表示检测出来的句子长度为2，则需要分成两段
        temp_split_list = [item[:indexes[0] + 1], item[indexes[0] + 1:]]
        csv_writer1.writerow([ids, item, indexes, temp_split_list])
    else:  # 如果返回的长度大于1，则需要分成几段
        indexes = [value + 1 for value in indexes]  # 在每一个本身的值上面加1
        print(indexes)
        new_indexes = sorted(indexes.extend(0))  # 排序后的索引列表，以0开头，最后一个切分位置结尾
        start, end = tee(item)
        split_list = [item[i:j] for i, j in zip(start, end)]
        split_list.append(item[indexes[-1]:])
        csv_writer1.writerow([ids, item, indexes, split_list])
    print(item)
    print('%s 判断处理完成' % ids)

f1.close()
f2.close()
f3.close()

# special_parse=['a.k.a.','e.g.','such as','aka','']
# 除此之外，还需要处理像(1)(2)这样的情况。
# 一些特定短语：
#     .. (dot dot)
#     . (dot)
#     cd ..
