import json

import numpy
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

def generate_adj_normalize(edge_index, num_items):
    row = np.zeros(len(edge_index), dtype=np.int32)
    col = np.zeros(len(edge_index), dtype=np.int32)

    cursor = 0
    for pair in edge_index:
        row[cursor] = pair[0]
        col[cursor] = pair[1]
        cursor += 1
    adj = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(num_items, num_items), dtype=np.float32)

    '''
    # 将矩阵转为对称矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = row_normalize_adj(adj)
    #D = np.array(adj.sum(axis=1)).reshape(-1)
    #L = sp.csr_matrix((D, (np.arange(num_items), np.arange(num_items))), shape=(num_items, num_items)) - adj
    L = sp.eye(adj.shape[0]) - adj
    '''

    # 假设邻接矩阵A已知
    # 构建度数矩阵D和对称归一化的邻接矩阵DAD
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    D = np.asarray(adj.sum(axis=1)).squeeze()
    print("D:",sum(D)/2)
    D_sqrt = np.sqrt(D)
    D_sqrt_inv = np.reciprocal(D_sqrt, where=D_sqrt != 0)
    D_sqrt_inv_mat = sp.csr_matrix(np.diag(D_sqrt_inv))
    DAD = D_sqrt_inv_mat.dot(adj).dot(D_sqrt_inv_mat)

    # 构建对称归一化的拉普拉斯矩阵L_sym
    L = sp.eye(adj.shape[0]) - DAD

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
parser.add_argument('--dataset', type=str, default='Toys')

args = parser.parse_args()

if __name__ == '__main__':

    print('Dataset: {}'.format(args.dataset))
    dataset = args.dataset

    sim_edge_index = []
    com_edge_index = []
    #features = []
    dataset_path = './data_preprocess/data/' + str(dataset) + '.json'
    with open(dataset_path, encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line: # 到 EOF，返回空字符串，则终止循环
                break
            js = json.loads(line)
            edge_list = js['edge']
            for i in edge_list:
                if (i['edge_type'] == 0 or i['edge_type'] == 2):
                    sim_edge_index.append([i['src_id'], i['dst_id']])
                if (i['edge_type'] == 1 or i['edge_type'] == 3):
                    com_edge_index.append([i['src_id'], i['dst_id']])
            #features.append(list(js['uint64_feature'].values()))

    path = './dataset/processed_val/' + str(dataset) + '_price.npz'
    data = np.load(path, allow_pickle=True)
    #com_edge_index = data['com_edge_index']
    #sim_edge_index = data['sim_edge_index']

    features = data['features']
    path = './data_preprocess/embs/' + str(dataset) + '_embeddings.npz'
    data = np.load(path, allow_pickle=True)
    cid3_emb = data['cid3_emb']
    cid2_emb = data['cid2_emb']

    features = np.squeeze(np.array(features))
    cid3 = features[:, 1].astype(int)
    cid3_emb_feature = cid3_emb[cid3]
    cid2 = features[:, 0].astype(int)
    cid2_emb_feature = cid2_emb[cid2]

    features = np.concatenate((cid2_emb_feature, cid3_emb_feature, np.expand_dims(features[:, 2], 1)), axis=1)

    features_emb = np.squeeze(np.array(features))

    '''
    path = './data_preprocess2/embs/' + str(dataset) + '_img_embeddings.npz'
    data = np.load(path, allow_pickle=True)
    features_emb = data['img_emb']
    print(features_emb.shape)
    '''

    sim_edge_index = list(set(tuple(i) for i in sim_edge_index) - set(tuple(i) for i in com_edge_index))
    L = generate_adj_normalize(sim_edge_index, len(features_emb))

    sim_score = []
    for i in range(features_emb.shape[1]):
        x = features_emb[:,i]
        score_high = x.T.dot(L.dot(x))
        tem = x.T.dot(x)
        sim_score.append(round((score_high/tem), 4))

    com_score = []
    print("com edge:")
    L = generate_adj_normalize(com_edge_index, len(features_emb))
    for i in range(features_emb.shape[1]):
        '''
        x = np.expand_dims(features_emb[:,i], axis=1)
        score_high = (x.T.dot(L))
        score_high = score_high.dot(x)
        tem = x.T.dot(x)
        com_score.append(round((float(score_high/tem)), 4))
        '''
        x = features_emb[:,i]
        score_high = x.T.dot(L.dot(x))
        tem = x.T.dot(x)
        com_score.append(round((score_high/tem), 4))

    #print("sim score:", sim_score)
    #print("com score:", com_score)
    diff_score = [com_score[i]-sim_score[i] for i in range(len(com_score))]
    print("max diff socre:", max(diff_score), "index:", diff_score.index(max(diff_score)))
    lht = []
    lht_idx = []
    for i in range(len(diff_score)):
        if diff_score[i] > 0:
            lht.append(diff_score[i])
    print(len(lht))
    print(len(diff_score))
    print(sum(diff_score)/len(diff_score))
    print(sum(com_score)/len(com_score))
    print("node num:", len(features_emb))
    print("edge num:", len(com_edge_index))
    '''
    com_price = []
    price = features[:,3]
    lht = []
    for pair in com_edge_index:
        com_price.append([price[pair[0]], price[pair[1]]])
        lht.append(abs(price[pair[0]]-price[pair[1]]))
    com_price = np.array(com_price)
    print(com_price)
    lht =np.array(lht)
    print(np.mean(lht))
    print(np.std(lht))

    com_price = []
    price = features[:,3]
    lht = []
    for pair in sim_edge_index:
        com_price.append([price[pair[0]], price[pair[1]]])
        lht.append(abs(price[pair[0]]-price[pair[1]]))
    com_price = np.array(com_price)
    print(com_price)
    print(np.mean(lht))
    print(np.std(lht))
    '''

