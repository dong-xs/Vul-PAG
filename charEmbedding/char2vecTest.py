import chars2vec

c2v_model=chars2vec.load_model('eng_150')

words1=['There','might','be','a','scenario','in','which','this','allows','remote','attackers','to','bypass','intended','access','restrictions','.']

word_embedding1=c2v_model.vectorize_words(words1)

# print(word_embedding1.shape)
print(word_embedding1.shape)

'''
    明天总结一下，char embedding的输入和输出是什么，以及如何修改模型来调整模型的输出维度。
'''