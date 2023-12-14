from utils import *
import argparse

parser = argparse.ArgumentParser(description='GCO')
parser.add_argument('--dataset', type=str, default='Appliances')
parser.add_argument('--neg_num', type=int, default=100)
args = parser.parse_args()

if __name__ == '__main__':

    print('Dataset: {}'.format(args.dataset))
    dataset = args.dataset
    neg_num = args.neg_num

    path = './baseline_model/CIKM2020_DecGCN/preprocessing/data/'
    dataset_path = path + dataset + '.json'

    save_path = './dataset/processed_val/' + dataset + '_percent_price.npz'

    sim_dict = {}
    com_dict = {}
    sim_edge_index = []
    com_edge_index = []
    features = []

    with open(dataset_path, encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:  # 到 EOF，返回空字符串，则终止循环
                break
            js = json.loads(line)
            neighbors = js['neighbor']
            key_item = js['node_id']

            sim_list = list(neighbors["0"].keys()) + list(neighbors["2"].keys())
            com_list = list(neighbors["1"].keys()) + list(neighbors["3"].keys())
            if len(sim_list)>2:
                sim_dict[key_item] = sim_list
                for dst_id in sim_list:
                    sim_edge_index.append([key_item, dst_id])
            if len(com_list)>2:
                com_dict[key_item] = com_list
                for dst_id in com_list:
                    com_edge_index.append([key_item, dst_id])

            features.append(list(js['uint64_feature'].values()))

    features = np.squeeze(np.array(features))

    num_items = features.shape[0]
    feature_size = features.shape[1]
    train_com_edge_index, train_set, val_set, test_set = pair_construct_percent(com_dict, num_items, 2, neg_num)

    np.savez(save_path, features = features, sim_edge_index = sim_edge_index, com_edge_index = train_com_edge_index,
             train_set = train_set, val_set = val_set, test_set = test_set)
    print("finished")









