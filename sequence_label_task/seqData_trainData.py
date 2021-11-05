#encoding:utf-8
#本py文件的作用就是：将标注的数据组合成seq文件中train_data的形式，该函数的方法与org_LSTM中的dataset_get()一致

def dataset_get(filename):   #该函数用于构建训练集和测试集
    data = open(filename, 'r')
    content = data.readlines()
    data.close()

    indexes = [0]
    for i in range(len(content)):
        if content[i] == '\n':  # 找到每个位置为'\n'的索引
            indexes.append(i)

    indexes.append(-1)

    for value in range(2, len(indexes) - 1, 2):
        indexes[value] += 1

    sentence_label = []  # 用来存储每个分句后的结果
    for value in range(0, len(indexes) - 1, 2):
        sentence_label.append(content[indexes[value]:indexes[value + 1]])

    sent_length = len(sentence_label)
    # 接下来需要将里面的每个文本进行转换
    # 长度为句子长度，每个位置上的元素由两部分构成，即token序列和label序列构成
    train_data = [[] for index in range(sent_length)]

    for i in range(sent_length):
        temp_sent = []
        temp_label = []
        for value in sentence_label[i]:
            lists = value.strip().split('  ')
            temp_sent.append(lists[0])
            temp_label.append(lists[1])
        train_data[i] = (temp_sent, temp_label)
    return train_data

# trains_data=dataset_get('merge_label_train.txt')
# test_data=dataset_get('lqd_label_result_test.txt')

trains_data=dataset_get('../generate_data/train_data_zip.txt')
test_data=dataset_get('../generate_data/test_data_zip.txt')
print(test_data)