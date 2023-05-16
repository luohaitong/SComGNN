import json
import numpy as np
import scipy.sparse as sp
import torch
from  scipy.sparse.linalg import eigsh
import argparse

def generate_adj(edge_index, num_items):
    row = np.zeros(len(edge_index), dtype=np.int32)
    col = np.zeros(len(edge_index), dtype=np.int32)

    cursor = 0
    for pair in edge_index:
        row[cursor] = pair[0]
        col[cursor] = pair[1]
        cursor += 1
    adj = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(num_items, num_items), dtype=np.float32)

    # 将矩阵转为对称矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    '''
    # 有对角线
    adj = row_normalize_adj(adj + sp.eye(adj.shape[0]))
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    '''
    #adj = row_normalize_adj(adj)
    D = np.array(adj.sum(axis=1)).reshape(-1)
    L = sp.csr_matrix((D, (np.arange(num_items), np.arange(num_items))), shape=(num_items, num_items)) - adj
    #L = sp.csr_matrix((D, (np.ones(num_items), np.ones(num_items))), shape=(num_items, num_items)) -  adj
    '''
    eigval, eigvec = eigsh(L, k=5, which='SM')
    U = eigvec.T
    '''

    return L

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

def fenjie(A, num_items, edge_index):
    row = np.zeros(len(edge_index), dtype=np.int32)
    col = np.zeros(len(edge_index), dtype=np.int32)

    cursor = 0
    for pair in edge_index:
        row[cursor] = pair[0]
        col[cursor] = pair[1]
        cursor += 1

    # 计算拉普拉斯矩阵 L 和特征向量 U
    D = np.array(A.sum(axis=1)).reshape(-1)
    print()
    L = sp.csr_matrix((D, (row, col)), shape=(num_items, num_items)) -A
    eigval, eigvec = eigsh(L, k=5, which='SM')
    U = eigvec.T
    '''
    # 计算频率分量
    X_freq = np.dot(U, X)

    print("邻接矩阵 A：\n", A.toarray())
    print("特征矩阵 X：\n", X)
    print("频率分量 X_freq：\n", X_freq)
    '''
    return eigval


parser = argparse.ArgumentParser(description='GCO_analyze')
parser.add_argument('--dataset', type=str, default='Toys_and_Games')

args = parser.parse_args()

if __name__ == '__main__':

    print('Dataset: {}'.format(args.dataset))
    dataset = args.dataset

    sim_edge_index = []
    com_edge_index = []
    features = []
    dataset_path = './dataset/' + str(dataset) + '.json'
    with open(dataset_path, encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line: # 到 EOF，返回空字符串，则终止循环
                break
            js = json.loads(line)
            edge_list = js['edge']
            for i in edge_list:
                if i['edge_type'] == 0:
                    sim_edge_index.append([i['src_id'], i['dst_id']])
                if i['edge_type'] == 1:
                    com_edge_index.append([i['src_id'], i['dst_id']])
            features.append(list(js['uint64_feature'].values()))

    features = np.squeeze(np.array(features))

    L = generate_adj(sim_edge_index, len(features))

    sim_score = []
    for i in range(features.shape[1]):
        x = features[:,i]
        score_high = x.T.dot(L.dot(x))
        tem = x.T.dot(x)
        sim_score.append(round((score_high/tem), 4))

    com_score = []
    L = generate_adj(com_edge_index, len(features))
    for i in range(features.shape[1]):
        x = features[:,i]
        score_high = x.T.dot(L.dot(x))
        tem = x.T.dot(x)
        com_score.append(round((score_high/tem), 4))

    print("sim score:", sim_score)
    print("com score:", com_score)
