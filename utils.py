import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import KBinsDiscretizer

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

def load_dataset(dataset, price_n_bins = 20):

    #load data
    data = np.load("./data_preprocess/processed/{}.npz".format(dataset), allow_pickle=True)

    features = data['features']
    com_edge_index = data['com_edge_index']
    train_set = data['train_set']
    val_set = data['val_set']
    test_set = data['test_set']

    #load category embedding
    data = np.load("./data_preprocess/embs/{}_embeddings.npz".format(dataset), allow_pickle=True)
    cid3_emb = data['cid3_emb']
    cid2_emb = data['cid2_emb']

    features = np.squeeze(np.array(features))
    cid3 = features[:, 1].astype(int)
    cid3_emb_feature = cid3_emb[cid3]
    cid2 = features[:, 0].astype(int)
    cid2_emb_feature = cid2_emb[cid2]

    category_emb = np.concatenate((cid2_emb_feature, cid3_emb_feature), axis=1)

    #discretize the continuous price to bins using equal-depth binnin
    est = KBinsDiscretizer(n_bins=price_n_bins, encode="ordinal")
    price = features[:, 2][:, np.newaxis]
    est.fit(price)
    price_bin = est.transform(price).squeeze()

    return category_emb, price_bin, com_edge_index, train_set, val_set, test_set

def generate_adj(edge_index, num_items):
    '''generate sparse adj'''
    row = np.zeros(len(edge_index), dtype=np.int32)
    col = np.zeros(len(edge_index), dtype=np.int32)

    cursor = 0
    for pair in edge_index:
        row[cursor] = pair[0]
        col[cursor] = pair[1]
        cursor += 1
    adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(num_items, num_items), dtype=np.float32)

    #Turn the matrix into a symmetric matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj_norm = row_normalize_adj(adj)
    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)
    print("Finish generating adjacent matrix")

    return adj_norm

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