# encoding:utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

content = pd.read_csv('predict.csv')

cvdids = content['ID']
descriptions = content['description']
tokens = [eval(token) for token in content['tokens']]
labels = [eval(label) for label in content['predict_label']]
vectors = [np.array(eval(vector)) for vector in content['embedding_vector']]  # 最外层是一个列表，列表中的每一个位置上都是一个二维矩阵
length = len(cvdids)

B_VAV_train_test_index = []  # 存储用于训练B-VAV的数据集，每一个存储的是索引值
B_VAT_train_test_index = []
B_VR_train_test_index = []
B_VAV_predict_index = []  # 存储需要预测B-VAV的数据集
B_VAT_predict_index = []
B_VR_predict_index = []
# 全部过程都用索引值来记就ok

for i in range(length):
    if 'B-VAV' in labels[i] and 'B-VN' in labels[i] and 'B-VAT' in labels[i] and 'B-VR' in labels[i]:
        B_VAV_train_test_index.append(i)
        B_VAT_train_test_index.append(i)
        B_VR_train_test_index.append(i)
    elif 'B-VAV' not in labels[i] and 'B-VN' in labels[i] and 'B-VAT' in labels[i] and 'B-VR' in labels[i]:
        B_VAV_predict_index.append(i)
    elif 'B-VAV' in labels[i] and 'B-VN' in labels[i] and 'B-VAT' not in labels[i] and 'B-VR' in labels[i]:
        B_VAT_predict_index.append(i)
    elif 'B-VAV' in labels[i] and 'B-VN' in labels[i] and 'B-VAT' in labels[i] and 'B-VR' not in labels[i]:
        B_VR_predict_index.append(i)


# print(len(B_VAV_train_test_index))
# print(len(B_VAV_predict_index))
# print(len(B_VAT_train_test_index))
# print(len(B_VAT_predict_index))
# print(len(B_VR_train_test_index))
# print(len(B_VR_predict_index))

# 先从训练集中把每个标签的类别确定出来，那么目前就是根据当前的向量将每一个训练集中该属性的向量表示出来，先以B_VAT为例
def function(start_label, end_label, train_test_set, label_set, vector_set):
    result_vector = []  # 用于存储所有节点的向量值，外层为列表，每个列表位置上为一个918维的array，列表长度与train_test_set一致
    for item in train_test_set:
        temp_vector = np.zeros(vector_set[item].shape[1])  # 生成与token维度等长的全0数组
        start_index = []
        end_index = []
        for j in range(len(label_set[item]) - 1):
            if labels[item][j] == start_label:  # 起始位置
                start_index.append(j)
            elif labels[item][j] == end_label and labels[item][j + 1] != end_label:  # 末尾位置
                end_index.append(j)
        # 还需要将最后一个位置考虑进去
        if labels[item][-1] == start_label:
            start_index.append(len(label_set[item]) - 1)
        if labels[item][-1] == end_label:
            end_index.append(len(label_set[item]) - 1)

        if len(start_index) == 1 and 1 == len(end_index):  # 如果只有一个VAT，且属性长度大于1时
            for value in range(start_index[0], end_index[0] + 1):
                temp_vector += vector_set[item][value]
            temp_vector /= end_index[0] - start_index[0] + 1
        elif len(start_index) == 1 and len(end_index) == 0:  # 如果只有一个VAT，且属性长度刚好等于1时
            temp_vector = vector_set[item][start_index[0]]
        elif len(start_index) > 1 and len(end_index) == 0:  # 如果有多个VAT，且每个的长度均等于1时
            for value in start_index:
                temp_vector += vector_set[item][value]
            temp_vector /= len(start_index)
        elif len(start_index) == len(end_index) and len(start_index) > 1:  # 如果有多个VAT，且每一个的长度都大于1，则选择长度更短的那个VAT作为最终结果
            min_gap = end_index[0] - start_index[0]
            start = start_index[0]
            end = end_index[0]
            for i in range(1, len(start_index)):
                if min_gap > end_index[i] - start_index[i]:
                    min_gap = end_index[i] - start_index[i]
                    start = start_index[i]
                    end = end_index[i]
            for value in range(start, end + 1):
                temp_vector += vector_set[item][value]
            temp_vector /= min_gap + 1
        result_vector.append(temp_vector)
    return result_vector


B_VAT_cluster_vectors=np.array(function('B-VAT','I-VAT',B_VAT_train_test_index,labels,vectors))
# B_VAV_cluster_vectors = np.array(function('B-VAV', 'I-VAV', B_VAV_train_test_index, labels, vectors))
# B_VR_cluster_vectors=np.array(function('B-VR','I-VR',B_VR_train_test_index,labels,vectors))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

ss = []
ch = []
dbs = []
for k in range(3, 16):  # 总共选择3-15个聚类中心，选择轮廓系数或CH分数最小的
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(B_VAT_cluster_vectors)
    predict_labels = kmeans.labels_

    ss_score = silhouette_score(B_VAT_cluster_vectors, predict_labels, sample_size=len(predict_labels),
                                metric='euclidean')  # 轮廓系数
    ch_score = calinski_harabasz_score(B_VAT_cluster_vectors, predict_labels)  # CH系数
    dbs_score = davies_bouldin_score(B_VAT_cluster_vectors, predict_labels)  # DBS系统
    ss.append(ss_score)
    ch.append(ch_score)
    dbs.append(dbs_score)

    print("the k of cluster is:", k)
    print("轮廓系数值为：", ss_score)  # 值越大越好
    print('CH系数值为：', ch_score)  # 值越大越好
    print('DBS系数值为：', dbs_score)  # 值越小越好

    mark = ['or', '4b', 'Dg', '*k', 'hr', '+c', 'sm', 'dy', '<w', 'pk', 'vr', 'Hb', '2g', '1k', '3r']

    j = 0
    for i in predict_labels:
        plt.plot([B_VAT_cluster_vectors[j:j + 1, 0]], [B_VAT_cluster_vectors[j:j + 1, 1]], mark[i], markersize=3)
        j += 1
    plt.title("%d cluters" % k)
    plt.show()

print('the ss value list:', ss)
print('the ch vlaue list:', ch)
print('the dbs value list:', dbs)

'''
针对B_VAT_cluster_vectors数据：
the ss value list: [0.4129364494844961, 0.4003827572092751, 0.4059112429064561, 0.13597834567302391, 0.14227655357284963, 0.14813559937681947, 0.1434925994171317, 0.1448506554141691, 0.14567316710823047, 0.14803441099809275, 0.09979697934965676, 0.1056952764100869, 0.10946996812941877]
the ch vlaue list: [493.14682658417524, 386.68419192510527, 338.03554315133044, 296.84528136931476, 264.95242001091816, 244.98331054641883, 232.23015324720862, 214.16990927906937, 198.23054303224976, 187.22584327066613, 174.03286480942006, 167.77161176364504, 159.37418280450166]
the dbs value list: [1.6993694117922908, 1.5956230628227894, 1.4115453153769366, 2.06424327965119, 2.1769367175732803, 2.0115335227171216, 1.9013044358885247, 1.9898204268064277, 2.256920608451203, 2.139186857125436, 2.499031173656948, 2.331544460832429, 2.4680351713924797]
上面表示各个聚类评价指标在不同K[3到10]值时的情况，可以发现：
    （1）ss是要越大越好，当k=5时，取到最大，然后急剧下降，且一直处于动荡阶段
    （2）ch也是要越大越好，但CH呈现单调递减情况，暂时不明白什么情况
    （3）dbs是要越小越好，当k=5时，取到最小，然后急剧上升，直至动荡，
'''

'''
针对B_VAV_cluster_vectors数据：
the ss value list: [0.1033569940778804, 0.12217661120963011, 0.12129369998433964, 0.10786924900083661, 0.11223350143099765, 0.11454340565442447, 0.1235371486217183, 0.11736277512165484, 0.11907075439619595, 0.12455474140798395, 0.12286263463855693, 0.11976797299305213, 0.12111453387237325]
the ch vlaue list: [147.23171878731188, 137.73610170832413, 123.76943526700728, 110.34721497510299, 99.48403411194523, 93.56995494292674, 87.99560082026724, 83.44044210939919, 78.78787593175637, 75.30248032691983, 71.34087455112667, 67.7622384818715, 64.45586276521877]
the dbs value list: [2.4586400200081333, 2.3394553564381324, 2.422871632085294, 2.9327962892393393, 2.633229972867967, 2.4208094648879674, 2.4410178142571755, 2.472726222198495, 2.4715288179798436, 2.4257710733877165, 2.4006424675906137, 2.5472248225397047, 2.5745063928189214]
上面表示各个聚类评价指标在不同K[3到15]值时的情况，可以发现：
    （1）ss是要越大越好，当k=4、9、12时，均呈现局部最大，且一直处于动荡阶段
    （2）ch也是要越大越好，但CH呈现单调递减情况，暂时不明白什么情况
    （3）dbs是要越小越好，当k=4、8、13时，均呈现局部最小，且一直处于动荡阶段
'''

'''
针对B_VR_cluster_vectors数据：
the ss value list: [0.1985893652916407, 0.2376433505166154, 0.2566756096527197, 0.27113565367612236, 0.28137938461283385, 0.29011723508600606, 0.2802514128458501, 0.2862198193224743, 0.22373294782180453, 0.24556470792041057, 0.24940996985700248, 0.24912696351957422, 0.23487175693253928]
the ch vlaue list: [304.25654469408056, 295.71502085357537, 264.39929889741086, 245.93699530682062, 229.51051564870684, 216.22901130625323, 209.63458464968775, 198.18330813822092, 188.46898767977356, 177.2259301655437, 170.68983068808993, 163.45323929568582, 155.75743891517544]
the dbs value list: [1.8017511131237403, 1.7411419583077752, 1.6932768975450656, 1.9341814898810423, 1.707735608116021, 1.84501861406069, 1.6909887312323522, 1.6387353989991797, 1.9842505502816357, 2.1291726975395937, 1.9838286326044907, 2.004375973809314, 2.092534475874385]
上面表示各个聚类评价指标在不同K[3到15]值时的情况，可以发现：
    （1）ss是要越大越好，当k=8时，取到最大，且一致处于正常状态
    （2）ch也是要越大越好，但CH呈现单调递减情况，暂时不明白什么情况
    （3）dbs是要越小越好，当k=6、11时，取到最小，然后急剧上升，直至动荡
'''

# 图例说明：https://blog.csdn.net/Treasure99/article/details/106044114
