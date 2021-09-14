#encoding:utf-8
#bert的tensorflow安装可参照链接：https://cloud.tencent.com/developer/news/492066
#BERT启动命令：bert-serving-start -model_dir I:\cased_L-12_H-768_A-12 -num_worker=1

from bert_serving.client import BertClient
client=BertClient()
vectors=client.encode(['dog','cat','man'])

from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(vectors[1,:],vectors[2,:]))
