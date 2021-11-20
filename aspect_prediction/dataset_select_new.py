# encoding:utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

content = pd.read_csv('predict_1.csv')

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
            #根据文章《Dissecting Contextual Word Embeddings:Architecture and Representation》的说法，
            # 对于有间距的短语表示，可以直接拼接头部token的embedding和最后token的embedding
            temp_vector=vector_set[item][start].extend(vector_set[item][end])

        result_vector.append(temp_vector)
    return result_vector


B_VAT_cluster_vectors=np.array(function('B-VAT','I-VAT',B_VAT_train_test_index,labels,vectors))[:,:-150]
# B_VAV_cluster_vectors = np.array(function('B-VAV', 'I-VAV', B_VAV_train_test_index, labels, vectors))[:,:-150]    #只取前768的word embedding
# B_VR_cluster_vectors=np.array(function('B-VR','I-VR',B_VR_train_test_index,labels,vectors))[:,:-150]

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
取768+150维时：
针对B_VAT_cluster_vectors数据：
the ss value list: [0.4129364494844961, 0.4003827572092751, 0.4059112429064561, 0.13597834567302391, 0.14227655357284963, 0.14813559937681947, 0.1434925994171317, 0.1448506554141691, 0.14567316710823047, 0.14803441099809275, 0.09979697934965676, 0.1056952764100869, 0.10946996812941877]
the ch vlaue list: [493.14682658417524, 386.68419192510527, 338.03554315133044, 296.84528136931476, 264.95242001091816, 244.98331054641883, 232.23015324720862, 214.16990927906937, 198.23054303224976, 187.22584327066613, 174.03286480942006, 167.77161176364504, 159.37418280450166]
the dbs value list: [1.6993694117922908, 1.5956230628227894, 1.4115453153769366, 2.06424327965119, 2.1769367175732803, 2.0115335227171216, 1.9013044358885247, 1.9898204268064277, 2.256920608451203, 2.139186857125436, 2.499031173656948, 2.331544460832429, 2.4680351713924797]
上面表示各个聚类评价指标在不同K[3到10]值时的情况，可以发现：
    （1）ss是要越大越好，当k=5时，取到最大，然后急剧下降，且一直处于动荡阶段
    （2）ch也是要越大越好，但CH呈现单调递减情况，暂时不明白什么情况
    （3）dbs是要越小越好，当k=5时，取到最小，然后急剧上升，直至动荡
可以确定，在该情况下B-VAT的最优类别为5
'''

'''
取768+150维时：
针对B_VAV_cluster_vectors数据：
the ss value list: [0.1033569940778804, 0.12217661120963011, 0.12129369998433964, 0.10786924900083661, 0.11223350143099765, 0.11454340565442447, 0.1235371486217183, 0.11736277512165484, 0.11907075439619595, 0.12455474140798395, 0.12286263463855693, 0.11976797299305213, 0.12111453387237325]
the ch vlaue list: [147.23171878731188, 137.73610170832413, 123.76943526700728, 110.34721497510299, 99.48403411194523, 93.56995494292674, 87.99560082026724, 83.44044210939919, 78.78787593175637, 75.30248032691983, 71.34087455112667, 67.7622384818715, 64.45586276521877]
the dbs value list: [2.4586400200081333, 2.3394553564381324, 2.422871632085294, 2.9327962892393393, 2.633229972867967, 2.4208094648879674, 2.4410178142571755, 2.472726222198495, 2.4715288179798436, 2.4257710733877165, 2.4006424675906137, 2.5472248225397047, 2.5745063928189214]
上面表示各个聚类评价指标在不同K[3到15]值时的情况，可以发现：
    （1）ss是要越大越好，当k=4、9、12时，均呈现局部最大，且一直处于动荡阶段，总体来说，在12的位置取值最大
    （2）ch也是要越大越好，但CH呈现单调递减情况，暂时不明白什么情况
    （3）dbs是要越小越好，当k=4、8、13时，均呈现局部最小，且一直处于动荡阶段，总体来说，在5的位置取值最小
针对处于动荡的情况，其实应当考虑其波动范围，总体来说，在值为5到6时，波动范围最大
可以确定，在该情况下B-VAV的最优类别为5时
'''

'''
取768+150维时：
针对B_VR_cluster_vectors数据：
the ss value list: [0.1985893652916407, 0.2376433505166154, 0.2566756096527197, 0.27113565367612236, 0.28137938461283385, 0.29011723508600606, 0.2802514128458501, 0.2862198193224743, 0.22373294782180453, 0.24556470792041057, 0.24940996985700248, 0.24912696351957422, 0.23487175693253928]
the ch vlaue list: [304.25654469408056, 295.71502085357537, 264.39929889741086, 245.93699530682062, 229.51051564870684, 216.22901130625323, 209.63458464968775, 198.18330813822092, 188.46898767977356, 177.2259301655437, 170.68983068808993, 163.45323929568582, 155.75743891517544]
the dbs value list: [1.8017511131237403, 1.7411419583077752, 1.6932768975450656, 1.9341814898810423, 1.707735608116021, 1.84501861406069, 1.6909887312323522, 1.6387353989991797, 1.9842505502816357, 2.1291726975395937, 1.9838286326044907, 2.004375973809314, 2.092534475874385]
上面表示各个聚类评价指标在不同K[3到15]值时的情况，可以发现：
    （1）ss是要越大越好，当k=8时，取到最大，且一致处于正常状态，但后面又到了10，且10到11的震荡范围最大，因此可以认为8的情况为局部最优
    （2）ch也是要越大越好，但CH呈现单调递减情况，暂时不明白什么情况
    （3）dbs是要越小越好，当k=5、10时，取到最小，然后急剧上升，直至动荡
可以确定，在该情况下B-VR的最优类别为10
'''

'''
取768维时：
针对B_VAT_cluster_vectors数据：
the ss value list: [0.4025347984922777, 0.39007789360941036, 0.39690852316325526, 0.1364125773552467, 0.140252783868242, 0.142403161147974, 0.1356236689090812, 0.1368052328497632, 0.14343564077848342, 0.10319688005908406, 0.1047987288470604, 0.10142342597905565, 0.14573651654368328]
the ch vlaue list: [446.06720523565957, 362.638100308319, 307.5527269269785, 271.049158585679, 246.87096467278948, 225.6274943333996, 202.7892089983763, 194.62465880029302, 184.81926144362438, 171.24106069303954, 161.3603894547962, 150.87112057253657, 144.75575897678644]
the dbs value list: [1.735032589506739, 1.3857498188180926, 1.3844171297095988, 2.101909975838916, 1.989602392990667, 2.0197291933163473, 2.3325287551081297, 2.124113072872877, 2.0054759262821658, 2.2141232692874735, 2.1269920894283105, 2.6023382598892404, 2.3155949422291315]
上面表示各个聚类评价指标在不同K[3到15]值时的情况，可以发现：
    （1）ss是要越大越好，当k=5时，取到最大，且一致处于正常状态
    （2）ch也是要越大越好，但CH呈现单调递减情况，暂时不明白什么情况
    （3）dbs是要越小越好，当k=5时，取到最小，然后急剧上升，直至动荡
基于此，可以确定B_VAT的类别就是5类。。。
'''

'''
取768维时：
针对B_VAV_cluster_vectors数据：
the ss value list: [0.10369324466140313, 0.1189680094592512, 0.12071252933006016, 0.12893181398873713, 0.10359377139208252, 0.11173182021250673, 0.11446107656518781, 0.11088762004437504, 0.11665252451360272, 0.09236436752365298, 0.1163783261447677, 0.0902450444916512, 0.10008803301633332]
the ch vlaue list: [151.26004562818298, 137.20998516180478, 122.9839158106348, 109.81491149784209, 98.14271197627455, 93.13669536456189, 85.59155206754237, 82.35901583640624, 78.25234538272933, 73.21749239697495, 69.67062670121972, 66.41418352143512, 63.710940965197]
the dbs value list: [2.5433658884553463, 2.3153362245144686, 2.5078473936366583, 2.135052219038468, 2.621550659976475, 2.4196020458717364, 2.6807143846717287, 2.4429911173199956, 2.438689958667776, 2.5375365973075534, 2.472886807706295, 2.6123673122286957, 2.464830443809476]
上面表示各个聚类评价指标在不同K[3到15]值时的情况，可以发现：
    （1）ss是要越大越好，当k=6时，取到最大，且一致处于正常状态
    （2）ch也是要越大越好，但CH呈现单调递减情况，暂时不明白什么情况
    （3）dbs是要越小越好，当k=6时，取到最小，然后急剧上升，直至动荡
基于此，可以确定B_VAV的类别就是6类。。。
'''

'''
取768维时：
针对B_VR_cluster_vectors数据：
the ss value list: [0.22434478935095672, 0.23778128069256496, 0.25555982558112367, 0.26114969465182813, 0.27863139540108206, 0.2653260214015171, 0.2731377823851921, 0.21109328601158647, 0.23657410618605376, 0.18834072258670057, 0.2452504855461205, 0.24838443179306305, 0.22301294610122052]
the ch vlaue list: [309.9427376445965, 300.92006048085597, 268.962254209879, 235.09145904617628, 231.13489214137527, 218.40255506211628, 207.8661101297152, 196.05166570544944, 187.4098660993206, 172.30307276431225, 167.9420711151001, 160.05698520564346, 154.09004893069178]
the dbs value list: [2.049446419478994, 1.7241275416411697, 1.7240900678727051, 1.809724286828681, 1.7638299490046034, 1.7091419970967232, 1.7269032628921175, 1.8883046432821078, 1.9350774459359619, 1.7829587823245363, 2.1685842231032084, 2.1033358300631098, 2.017901520151329]
上面表示各个聚类评价指标在不同K[3到15]值时的情况，可以发现：
    （1）ss是要越大越好，当k=7时，取到最大，且一致处于正常状态，但到了9时达到了最大，且9到10的动荡最大，因此有可能7为局部最优
    （2）ch也是要越大越好，但CH呈现单调递减情况，暂时不明白什么情况
    （3）dbs是要越小越好，当k=8时，取到最小，然后急剧上升，直至动荡
基于此，可以确定B_VR的类别就是8或9
'''
# 图例说明：https://blog.csdn.net/Treasure99/article/details/106044114
