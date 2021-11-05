#encoding:utf-8
'''
    程序作用：返回了每个token的对应head及其index值
'''
import spacy

spacy_nlp=spacy.load('en_core_web_sm')

example='Cross-site scripting (XSS) vulnerability in Textpattern CMS before 4.5.7 allows remote attackers to inject arbitrary web script or HTML via the PATH_INFO to setup/index.php.'

def returnHeadList(sentence):
    docs=spacy_nlp(sentence)
    tokenList=[]
    tokenHeadList=[]
    for token in docs:
        tokenList.append(token.text)
        tokenHeadList.append(token.head.text)
    tokenHeadIndex=[tokenList.index(tokens) for tokens in tokenHeadList]
    return tokenList,tokenHeadList,tokenHeadIndex