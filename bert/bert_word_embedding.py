#encoding:utf-8
#bert的tensorflow安装可参照链接：https://cloud.tencent.com/developer/news/492066
#BERT启动命令：bert-serving-start -model_dir cased_L-12_H-768_A-12 -num_worker=1
#该模型参数的意思是12个layers，768的hidden-dim，12个head的attention
#在宿舍的电脑上跑不了bert程序，一跑就出错，只有尝试将bert放到服务器上跑试试看。。。

#推荐查看使用huggingface bert来进行微调：https://www.bilibili.com/video/BV1Dz4y1d7am

from bert_serving.client import BertClient
client=BertClient()
vectors=client.encode(['dog','cat','man'])

print(vectors)