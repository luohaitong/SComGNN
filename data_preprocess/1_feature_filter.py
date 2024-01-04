import sys

data_name = sys.argv[1]


#meta_file = open('../baseline_model/CIKM2020_DecGCN/preprocessing/raw_data/meta_{}.json'.format(data_name)).readlines()
meta_file = open('../raw_data/meta_{}.json'.format(data_name)).readlines()
out_file = open('./tmp/filtered_meta_{}.json'.format(data_name), 'w')

print("Filtering items with incomplete features...")

total_node_num = 0
feature_sets = [set() for _ in range(4)]

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

'''
for eachline in meta_file:
    data = eval(eachline)
    if len(data['category']) >= 4 and 'brand' in data and 'price' in data:
        cid1, cid2, cid3 = data['category'][1:4]
        bid = data['brand']
        price = data['price']
        if price== "":
            continue
        if is_float(price[1:]):
            price = float(price[1:])
        else:
            continue
        features = [cid2, cid3, bid, price]
        for i in range(len(features)):
            feature_sets[i].add(features[i])
        out_file.write(eachline)
        total_node_num += 1
'''

for eachline in meta_file:
    data = eval(eachline)
    if len(data['category']) >= 4 and 'price' in data:
        cid1, cid2, cid3 = data['category'][1:4]
        #bid = data['brand']
        price = data['price']
        if price== "":
            continue
        if is_float(price[1:]):
            price = float(price[1:])
        else:
            continue
        features = [cid2, cid3, price]
        for i in range(len(features)):
            feature_sets[i].add(features[i])
        out_file.write(eachline)
        total_node_num += 1

'''
for eachline in meta_file:
    data = eval(eachline)
    if len(data['category']) >= 4 and 'imageURLHighRes' in data:
        cid1, cid2, cid3 = data['category'][1:4]
        #bid = data['brand']
        imageurl = data['imageURLHighRes']
        if len(imageurl)==0:
            continue
        imageurl = imageurl[0]
        features = [cid2, cid3, imageurl]
        for i in range(len(features)):
            feature_sets[i].add(features[i])
        out_file.write(eachline)
        total_node_num += 1
'''
for i in range(len(features)):
    print(len(feature_sets[i]))
print('Total node num is {}'.format(total_node_num))

