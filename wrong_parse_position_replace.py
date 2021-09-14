'''
    该py文件的作用是


'''

import pandas as pd

data=pd.read_csv('wrong_parse_sentence.csv')

ids=data['id']
content=data['content']
parse_result=data['deps_result']