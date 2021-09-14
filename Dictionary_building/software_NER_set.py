# encoding:utf-8

'''
    该py文件的作用就是根据爬取到的CPE内容，构建每一个cve的受影响软件供应商、受影响软件名称、受影响软件版本
    CPE单条的书写规则：cpe:/<part>:<vendor>:<product>:<version>:<update>:<edition>:<language>
    CPE书写顺序规则：按顺序或逆序排列。
    输入：每一条cve对应的所有CPE内容
    输出：每一个供应商对应每一个产品的最新版本
    CPE抽取难点：
        1、每一条CPE的书写规则是固定的，但是其排列顺序不确定是顺序还是逆序
        2、因为需要找到所有vendor和product以及version，而无法确定在何时vendor、product会发生变化
        3、version会有不同的表现形式
'''

import pandas as pd

data = pd.read_csv('cpe_info.csv', names=['id', 'content', 'cveid'])

cpe_content = data['content']
cveids = data['cveid']

def get_vendor_product_version(sum_conts):
    vendor_list = []  # 需要记录每一个供应商的名字
    vendor_index = []  # 记录每一个供应商名字出现变化的索引位置
    product_list = []  # 需要记录每一个产品的名字及其出现变化的索引位置
    product_index = []  # 记录每一个产品名字出现变化的索引位置
    version_list = []  # 需要记录每一个产品所变化对应位置的版本
    version_index = []  # 记录每一个版本号出现变化的索引位置
    i = 0

    for conts in sum_conts:
        split_list = conts['cpe'].split(':')
        up_version = conts['up_version'].split()[-1] if conts['up_version'] != '' else ''
        vendor = split_list[3]
        product = split_list[4]
        version = split_list[5]

        if up_version != '':
            version = up_version  # 如果up_version不为空，则使用up_version替换掉原来的version
            if version not in version_list:
                version_list.append(version)
                version_index.append(i)
        else:
            if version not in version_list and version != '-':
                version_list.append(version)
                version_index.append(i)

        if vendor not in vendor_list and vendor != '':
            vendor_list.append(vendor)
            vendor_index.append(i)
        if product not in product_list and product != '':
            product_list.append(product)
            product_index.append(i)

        i = i + 1

    # 至此得到了每一个cve中受影响的供应商及其软件有哪些，以及这些软件的版本变化点在哪里。
    # 接下来就应该将这些值进行组合成最优的输出
    # 因为暂时没有这种最优组合数据的要求，因此可以先不考虑实现
    # if len(vendor_list)==1:         #若供应商长度为1
    #     if len(product_list)==1:    #若产品长度为1
    #         if len(version_list)==1:   #若版本号长度为1
    #             return [vendor_list[0],product_list[0],version_list[0]]
    #         else:           #若版本号长度不为1，则判断第一个版本和最后一个版本的大小即可，直接选择比较值最大的版本作为最终输出
    #             max_version=version_list[0] if version_list[0]>version_list[-1] else version_list[-1]
    #             return [vendor_list[0],product_list[0],max_version]
    #     else:   #若产品长度不为1，则需要基于产品进行切分，

    return vendor_list, vendor_index, product_list, product_index, version_list, version_index

import csv

headers = ['id', 'content', 'vendor', 'product', 'version']  # 未完全解析表的表头
f = open('product_version_NER_set.csv', 'w', encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(headers)

if __name__ == '__main__':
    # 接下来是将数据写入文件，写入格式为：id、vendor、product、version
    for ids, items in zip(cveids, cpe_content):
        items = eval(items)  # 将每一条数据转换成包含多个类似于“字典”字符串的列表，一个“字典结构”就是一个字典。即通过一次eval操作，实现了外层列表转换以及内层字典的转换
        vendors, vendors_index, products, products_index, versions, versions_index = get_vendor_product_version(items)
        csv_writer.writerow([ids, items, vendors, products, versions])
        print('%s 处理并写入完成' % ids)
    f.close()
