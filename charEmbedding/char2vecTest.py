import chars2vec

c2v_model=chars2vec.load_model('eng_150')

words1=['There','might','be','a','scenario','in','which','this','allows','remote','attackers','to','bypass','intended','access','restrictions','.']

word_embedding1=c2v_model.vectorize_words(words1)

# print(word_embedding1.shape)
print(word_embedding1.shape)

'''
    �����ܽ�һ�£�char embedding������������ʲô���Լ�����޸�ģ��������ģ�͵����ά�ȡ�
'''