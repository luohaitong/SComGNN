import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
import scipy.io as sio
import random
import pandas as pd
import copy as cp
from torch.utils.data import Dataset
from sklearn.preprocessing import KBinsDiscretizer
import json

###############################################
# Forked from GRAND-Lab/CoLA                  #
###############################################

def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret

# Process a (subset of) a TU dataset into standard form
def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))
       
    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))
    
    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# Forked from tkipf/gcn                       #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def row_normalize_adj(adj):
    """Row-normalize feature matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_feature(mx):
	"""
		Row-normalize sparse matrix
		Code from https://github.com/williamleif/graphsage-simple/
	"""
	rowsum = np.array(mx.sum(1)) + 0.01
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adj_to_dict(adj,hop=1,min_len=8):
    adj = np.array(adj.todense(),dtype=np.float64)
    num_node = adj.shape[0]
    # adj += np.eye(num_node)

    adj_diff = adj
    if hop > 1:
        for _ in range(hop - 1):
            adj_diff = adj_diff.dot(adj)


    dict = {}
    for i in range(num_node):
        dict[i] = []
        for j in range(num_node):
            if adj_diff[i,j] > 0:
                dict[i].append(j)

    final_dict = dict.copy()

    for i in range(num_node):
        while len(final_dict[i]) < min_len:
            final_dict[i].append(random.choice(dict[random.choice(dict[i])]))
    return dict

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    data = sio.loadmat("./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    labels = np.squeeze(np.array(data['Class'],dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels,num_classes)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[ : num_train]
    idx_val = all_idx[num_train : num_train + num_val]
    idx_test = all_idx[num_train + num_val : ]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels

def adj_to_dgl_graph(adj):
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph

def generate_rwr_subgraph(dgl_graph, subgraph_size):
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size*3)
    subv = []

    for i,trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)

    return subv

def generate_train_pair(train_mask, labels, PA_level, epoch):

    np.random.seed(epoch)

    item_id_pair = []
    cat_pair = []
    PA_pair = []
    train_labels = labels[train_mask]
    train_a_item = train_mask[train_labels == 1]
    train_n_item = train_mask[train_labels == 0]

    n_index = np.random.choice(range(len(train_n_item)), size=len(train_a_item))
    for i in range(len(n_index)):
        item1 = train_a_item[i]
        item2 = train_n_item[n_index[i]]
        item_id_pair.append([item1, item2])
        cat_pair.append([labels[item1], labels[item2]])
        #PA_pair.append(torch.cat([PA_level[item1].reshape(1, -1), PA_level[item2].reshape(1, -1)], dim=0))
        PA_pair.append(np.concatenate((np.array(PA_level[item1].cpu()).reshape(1, -1), np.array(PA_level[item1].cpu()).reshape(1, -1))
                                      , axis=0))

    item_id_pair = torch.tensor(item_id_pair)
    cat_pair = torch.tensor(cat_pair)
    PA_pair = torch.tensor(np.array(PA_pair))

    return item_id_pair, cat_pair, PA_pair

def generate_train_single(train_mask, labels, PA_level, epoch):

    np.random.seed(epoch)

    train_labels = labels[train_mask]
    train_a_mask = train_mask[train_labels == 1]
    train_n_mask = train_mask[train_labels == 0]
    train_n_mask_pick = np.random.choice((train_n_mask.cpu()), size=len(train_a_mask))

    train_mask_new = np.concatenate((train_n_mask_pick, train_a_mask.cpu()))
    item_id_pair = train_mask_new
    cat_pair = labels[train_mask_new].cpu()
    PA_pair = PA_level[train_mask_new].cpu()

    return item_id_pair, cat_pair, PA_pair

def pair_construct_1(item_dict, num_items, neg_num, random_seed=0):

    random.seed(random_seed)
    all_items = np.arange(num_items)
    train_triple_pair = []
    test_triple_pair = []

    for key_item in item_dict.keys():
        pos_list = item_dict[key_item]
        neg_list = list(set(all_items)-set(pos_list))
        neg_sample = random.sample(neg_list, len(pos_list)+neg_num-1)
        for i in range(len(pos_list)-1):
            train_triple_pair.append([int(key_item), int(pos_list[i]), int(neg_sample[i])])
        tem = [ int(key_item), int(pos_list[len(pos_list)-1])]
        tem.extend(neg_sample[len(pos_list)-1:])
        test_triple_pair.append(tem)

    train_triple_pair = np.array(train_triple_pair)
    test_triple_pair = np.array(test_triple_pair)

    return train_triple_pair, test_triple_pair

def pair_construct_2(item_dict, num_items, neg_num, random_seed = 1):

    random.seed(random_seed)
    all_items = np.arange(num_items)
    train_triple_pair = []
    val_triple_pair = []
    test_triple_pair = []
    for key_item in item_dict.keys():
        pos_list = item_dict[key_item]
        neg_list = list(set(all_items)-set(pos_list))
        neg_sample = random.sample(neg_list, len(pos_list)+ 2 * neg_num - 2)
        for i in range(len(pos_list)-2):
            train_triple_pair.append([int(key_item), int(pos_list[i]), int(neg_sample[i])])

        tem = [ int(key_item), int(pos_list[len(pos_list)-2])]
        tem.extend(neg_sample[len(pos_list)-2: len(pos_list)-2 + neg_num])
        val_triple_pair.append(tem)

        tem = [ int(key_item), int(pos_list[len(pos_list)-1])]
        tem.extend(neg_sample[len(pos_list)-2 + neg_num:])
        test_triple_pair.append(tem)

    train_triple_pair = np.array(train_triple_pair)
    val_triple_pair = np.array(val_triple_pair)
    test_triple_pair = np.array(test_triple_pair)

    return train_triple_pair, val_triple_pair, test_triple_pair

def pair_construct_3(item_dict, num_items, train_neg_num, test_neg_num, random_seed = 1):

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
        if len(pos_list)<3:
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
            #train_triple_pair.append([int(key_item), int(pos_list[i]), int(neg_sample[i])])

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

def pair_construct_percent(item_dict, num_items, train_neg_num, test_neg_num, random_seed = 1):

    random.seed(random_seed)
    all_items = np.arange(num_items)
    com_edge_index = []
    train_triple_pair = []
    val_triple_pair = []
    test_triple_pair = []
    for key_item in item_dict.keys():
        pos_list = item_dict[key_item]
        neg_list = list(set(all_items)-set(pos_list))
        if (int(len(pos_list)*0.6) < len(pos_list)-2):
            train_num = int(len(pos_list)*0.6)
            val_num = 1
            test_num = 1
        else:
            train_num = len(pos_list)-2
            val_num = int(len(pos_list)*0.2)
            test_num = len(pos_list)-train_num-val_num

        neg_sample = random.sample(neg_list, train_neg_num * train_num + test_neg_num * (val_num + test_num))
        for i in range(train_num):
            com_edge_index.append([int(key_item), int(pos_list[i])])
            tem = [int(key_item), int(pos_list[i])]
            tem.extend(neg_sample[train_neg_num * i: train_neg_num * (i+1)])
            train_triple_pair.append(tem)

        for i in range(train_num, train_num+val_num ):
            tem = [int(key_item), int(pos_list[i])]
            tem.extend(neg_sample[train_neg_num * train_num + (i-train_num)*test_neg_num:
                                  train_neg_num * train_num + (i-train_num+1)*test_neg_num])
            val_triple_pair.append(tem)

        for i in range(train_num+val_num, train_num+val_num+test_num):
            tem = [int(key_item), int(pos_list[i])]
            tem.extend(neg_sample[train_neg_num * train_num + (i-train_num)*test_neg_num:
                                  train_neg_num * train_num + (i-train_num+1)*test_neg_num])
            test_triple_pair.append(tem)

    train_triple_pair = np.array(train_triple_pair)
    val_triple_pair = np.array(val_triple_pair)
    test_triple_pair = np.array(test_triple_pair)

    return com_edge_index, train_triple_pair, val_triple_pair, test_triple_pair

def load_dataset(dataset):

    dataset_path = './data/' + dataset + '.json'
    sim_edge_index = []
    com_edge_index = []
    test_sim_edge_index = []
    test_com_edge_index = []
    features = []

    with open(dataset_path, encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:  # 到 EOF，返回空字符串，则终止循环
                break
            js = json.loads(line)
            edge_list = js['edge']
            neighbors = js['neighbor']
            
            for i in edge_list:
                if i['edge_type'] == 0:
                    sim_edge_index.append([i['src_id'], i['dst_id']])
                elif i['edge_type'] == 1:
                    com_edge_index.append([i['src_id'], i['dst_id']])
                elif i['edge_type'] == 2:
                    test_sim_edge_index.append((i['src_id', i['dst_id']]))
                elif i['edge_type'] == 3:
                    test_com_edge_index.append((i['src_id', i['dst_id']]))
                else:
                    print("edge type error")

            features.append(list(js['uint64_feature'].values()))

    features = np.squeeze(np.array(features))

    return features, sim_edge_index, com_edge_index, test_sim_edge_index, test_com_edge_index


def load_dataset2(dataset):
    dataset_path = './dataset/' + dataset + '.json'
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
            if len(sim_list)>1:
                sim_dict[key_item] = sim_list
                for dst_id in sim_list:
                    sim_edge_index.append([key_item, dst_id])
            if len(com_list)>1:
                com_dict[key_item] = com_list
                for dst_id in com_list:
                    com_edge_index.append([key_item, dst_id])

            features.append(list(js['uint64_feature'].values()))

    features = np.squeeze(np.array(features))

    return features, sim_dict, com_dict, sim_edge_index, com_edge_index

def load_dataset3(dataset):

    path = './dataset/processed_val/' + str(dataset) + '_contrastive.npz'
    data = np.load(path, allow_pickle=True)

    features = data['features']
    com_edge_index = data['com_edge_index']
    train_set = data['train_set']
    val_set = data['val_set']
    test_set = data['test_set']

    return features, com_edge_index, train_set, val_set, test_set

def load_dataset4(dataset):

    path = './dataset/processed_val/' + str(dataset) + '_percent_price.npz'
    data = np.load(path, allow_pickle=True)

    features = data['features']
    com_edge_index = data['com_edge_index']
    sim_edge_index = data['sim_edge_index']
    train_set = data['train_set']
    val_set = data['val_set']
    test_set = data['test_set']

    path = './dataset/processed_val/' + str(dataset) + '_embeddings.npz'
    data = np.load(path, allow_pickle=True)
    cid3_emb = data['cid3_emb']
    features = np.squeeze(np.array(features))
    features = features[:,1].astype(int)
    print(features.shape)
    features = cid3_emb[features]
    print(features.shape)
    return features, com_edge_index, sim_edge_index, train_set, val_set, test_set


def load_dataset4(dataset):
    path = './dataset/processed_val/' + str(dataset) + '_percent_price.npz'
    data = np.load(path, allow_pickle=True)

    features = data['features']
    com_edge_index = data['com_edge_index']
    sim_edge_index = data['sim_edge_index']
    train_set = data['train_set']
    val_set = data['val_set']
    test_set = data['test_set']

    return features, com_edge_index, sim_edge_index, train_set, val_set, test_set


def load_dataset_emb(dataset, price_n_bins = 20):
    path = './dataset/processed_val/' + str(dataset) + '_price.npz'
    data = np.load(path, allow_pickle=True)

    features = data['features']
    com_edge_index = data['com_edge_index']
    sim_edge_index = data['sim_edge_index']
    train_set = data['train_set']
    val_set = data['val_set']
    test_set = data['test_set']

    ''' load category embedding'''
    path = './data_preprocess/embs/' + dataset + '_embeddings.npz'
    data = np.load(path, allow_pickle=True)
    cid3_emb = data['cid3_emb']
    cid2_emb = data['cid2_emb']
    features = np.squeeze(np.array(features))
    cid3 = features[:, 1].astype(int)
    cid3_emb_feature = cid3_emb[cid3]
    cid2 = features[:, 0].astype(int)
    cid2_emb_feature = cid2_emb[cid2]

    '''price bin'''
    est = KBinsDiscretizer(n_bins=price_n_bins, encode="ordinal")
    price = features[:, 2][:, np.newaxis]
    est.fit(price)
    price_bin = est.transform(price).squeeze()

    #features_emb = np.concatenate((cid2_emb_feature, cid3_emb_feature, price_bin), axis=1)
    category_emb = np.concatenate((cid2_emb_feature, cid3_emb_feature), axis=1)
    return category_emb, price_bin, com_edge_index, sim_edge_index, train_set, val_set, test_set

def generate_adj(edge_index, num_items):

    #generate sparse adj
    row = np.zeros(len(edge_index), dtype=np.int32)
    col = np.zeros(len(edge_index), dtype=np.int32)

    cursor = 0
    for pair in edge_index:
        row[cursor] = pair[0]
        col[cursor] = pair[1]
        cursor += 1
    adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_items, num_items), dtype=np.float32)
    # 将矩阵转为对称矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #有对角线
    #adj_nor = row_normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_nor = row_normalize_adj(adj)
    adj_nor = sparse_mx_to_torch_sparse_tensor(adj_nor)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    print("finish generating adj")

    return adj_nor, adj

def generate_full_adj(train_mask, edge_index_list, labels, PA_list, character_list, num_items):
    if 1:
        #generate sparse adj
        adj_list = []
        mask_adj_list = []
        for i in range(len(edge_index_list)):
            edge_index = edge_index_list[i]
            row = np.zeros(len(edge_index) + 2 * num_items, dtype=np.int32)
            col = np.zeros(len(edge_index) + 2 * num_items, dtype=np.int32)

            if len(PA_list)>1:
                PA = PA_list[i]
            else:
                PA = PA_list[0]

            num_PA = int(max(PA)) + 1
            num_cats = int(max(labels)) + 1

            cursor = 0
            for pair in edge_index:
                row[cursor] = pair[0]
                col[cursor] = pair[1]
                cursor += 1

            for item_id in train_mask:
                row[cursor: cursor + 1] = int(item_id)
                col[cursor] = int(labels[item_id]) + num_items
                cursor += 1

            for item_id in range(num_items):
                row[cursor] = item_id
                col[cursor] = int(PA[item_id]) + num_items + num_cats
                cursor += 1


            adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_items+num_PA+num_cats, num_items+num_PA+num_cats),
                                dtype=np.float32)
            # 将矩阵转为对称矩阵
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            #无对角线
            mask_adj = adj.copy()
            mask_adj = row_normalize_adj(mask_adj)
            mask_adj = sparse_mx_to_torch_sparse_tensor(mask_adj)
            mask_adj_list.append(mask_adj)
            #有对角线
            adj = row_normalize_adj(adj + sp.eye(adj.shape[0]))
            adj = sparse_mx_to_torch_sparse_tensor(adj)
            adj_list.append(adj)
        print("finish generating adj")

    if 1:
        #generate cat adj
        num_cats = int(max(labels)) + 1
        row = np.zeros(num_items, dtype=np.int32)
        col = np.zeros(num_items, dtype=np.int32)

        cursor = 0
        for item_id in train_mask:
            col[cursor: cursor + 1] = int(item_id)
            row[cursor] = int(labels[item_id])
            cursor += 1
        cat_adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_cats, num_items), dtype=np.float32)
        # 将矩阵转为对称矩阵
        cat_adj = row_normalize_adj(cat_adj)
        cat_adj = sparse_mx_to_torch_sparse_tensor(cat_adj)

        print("finish generating cat adj")

    if 1:
        #generate PA adj
        PA_adj_list = []
        for PA in PA_list:
            num_PA = int(max(PA)) + 1
            row = np.zeros(num_items, dtype=np.int32)
            col = np.zeros(num_items, dtype=np.int32)

            cursor = 0
            for item_id in range(num_items):
                col[cursor: cursor + 1] = item_id
                row[cursor] = int(PA[item_id])
                cursor += 1
            PA_adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_PA, num_items), dtype=np.float32)
            # 将矩阵转为对称矩阵
            PA_adj = row_normalize_adj(PA_adj)
            PA_adj = sparse_mx_to_torch_sparse_tensor(PA_adj)
            PA_adj_list.append(PA_adj)

        print("finish generating PA adj")

    return mask_adj_list, adj_list, cat_adj, PA_adj_list


def pick_step(idx_train, y, edge_index, size):

    degree_all = torch.zeros_like(y)
    y_train = y[idx_train]
    for pair in edge_index:
        degree_all[pair[0]] += 1
        degree_all[pair[1]] += 1

    degree_train = degree_all[idx_train]
    #lf_train = (y_train.sum()-len(y_train))*y_train + len(y_train)
    pos_ratio = y_train.sum()/len(y_train)
    lf_train = cp.deepcopy(y_train)
    lf_train[lf_train == 1] = pos_ratio
    lf_train[lf_train != 1] = 1-pos_ratio
    smp_prob = degree_train/ lf_train
    return random.choices(idx_train, weights=smp_prob, k=size)

def pos_neg_split(nodes, labels):
    pos_nodes = []
    neg_nodes = []
    aux_nodes = cp.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(aux_nodes[idx])
        else:
            neg_nodes.append(aux_nodes[idx])
    return pos_nodes, neg_nodes

def generate_edge_label(train_mask, edge_index_list, labels):
    edge_train_mask = []
    edge_train_label = []
    for edge_index in edge_index_list:
        edge_train_mask_tmp = []
        for i in range(len(edge_index)):
            pair = edge_index[i]
            if pair[0] in train_mask and pair[1] in train_mask:
                edge_train_mask_tmp.append(i)
                if labels[pair[0]] == labels[pair[1]]:
                    edge_train_label.append(1)
                else:
                    edge_train_label.append(-1)
        edge_train_mask.append(edge_train_mask_tmp)

    return edge_train_mask, edge_train_label

class Train_Dataset(Dataset):

    def __init__(self, train_set):
        self.train_set = train_set

    def __getitem__(self, index):
        return self.train_set[index]

    def __len__(self):
        return len(self.train_set)

class Test_Dataset(Dataset):

    def __init__(self, item_id, PA_level):
        self.item_id = item_id
        self.PA_level = PA_level

    def __getitem__(self, index):
        return self.item_id[index], self.PA_level[index]

    def __len__(self):
        return len(self.item_id)