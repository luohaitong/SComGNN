import argparse
import json
import numpy as np
import random

parser = argparse.ArgumentParser(description='train_test_split')
parser.add_argument('--dataset', type=str, default='Appliances')
parser.add_argument('--train_neg_num', type=int, default=2)
parser.add_argument('--test_neg_num', type=int, default=100)
args = parser.parse_args()

def pair_construct(item_dict, num_items, train_neg_num, test_neg_num, random_seed = 1):

    random.seed(random_seed)
    all_items = np.arange(num_items)
    com_edge_index = []
    train_triple_pair = []
    val_triple_pair = []
    test_triple_pair = []

    for key_item in item_dict.keys():
        pos_list = item_dict[key_item]
        neg_list = list(set(all_items)-set(pos_list))
        #neg_sample = random.sample(neg_list, len(pos_list)+ 3 * neg_num - 2)
        neg_sample = random.sample(neg_list, train_neg_num * len(pos_list) + 2 * test_neg_num)
        if len(pos_list) < 3:
            #if the length of pos_list is smaller than 3, do not put it into the validation set and test set
            for i in range(len(pos_list)):
                tem = [int(key_item), int(pos_list[i])]
                com_edge_index.append([int(key_item), int(pos_list[i])])
                tem.extend(neg_sample[train_neg_num * i: train_neg_num * (i + 1)])
                train_triple_pair.append(tem)
            continue

        for i in range(len(pos_list)-2):
            tem = [int(key_item), int(pos_list[i])]
            com_edge_index.append([int(key_item), int(pos_list[i])])
            tem.extend(neg_sample[train_neg_num * i: train_neg_num * (i+1)])
            train_triple_pair.append(tem)

        tem = [ int(key_item), int(pos_list[len(pos_list)-2])]
        tem.extend(neg_sample[train_neg_num * len(pos_list):  train_neg_num * len(pos_list) + test_neg_num])
        val_triple_pair.append(tem)

        tem = [ int(key_item), int(pos_list[len(pos_list)-1])]
        tem.extend(neg_sample[train_neg_num * len(pos_list) + test_neg_num: ])
        test_triple_pair.append(tem)

    train_triple_pair = np.array(train_triple_pair)
    val_triple_pair = np.array(val_triple_pair)
    test_triple_pair = np.array(test_triple_pair)

    return com_edge_index, train_triple_pair, val_triple_pair, test_triple_pair

if __name__ == '__main__':

    print('Dataset: {}'.format(args.dataset))
    dataset = args.dataset

    path = './data/'
    dataset_path = path + dataset + '.json'

    save_path = '../dataset/processed_val/' + dataset + '.npz'

    com_dict = {}
    features = []

    with open(dataset_path, encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            js = json.loads(line)
            neighbors = js['neighbor']
            key_item = js['node_id']
            com_list = list(neighbors["1"].keys()) + list(neighbors["3"].keys())
            com_dict[key_item] = com_list
            features.append(list(js['uint64_feature'].values()))

    features = np.squeeze(np.array(features))
    num_items = features.shape[0]
    train_com_edge_index, train_set, val_set, test_set = pair_construct(com_dict, num_items, args.train_neg_num, args.test_neg_num)

    np.savez(save_path, features = features, com_edge_index = train_com_edge_index, train_set = train_set,
             val_set = val_set, test_set = test_set)
    print("train test split finished")









