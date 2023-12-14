import json
import numpy as np
import scipy.sparse as sp
import torch
from  scipy.sparse.linalg import eigsh
from  scipy.linalg import eigh
import matplotlib.pyplot as plt
import argparse
import os

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

def plot_frequency(features, L_sim, L_com, dataset_name, save_path):
    save_path = save_path + '/' + str(dataset_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    eigval_sim, eigvec_sim = eigh(L_sim.toarray())
    #eigvec_sim = eigvec_sim / np.linalg.norm(eigvec_sim, axis=0)

    eigval_com, eigvec_com = eigh(L_com.toarray())
    #eigvec_com = eigvec_com / np.linalg.norm(eigvec_com, axis=0)
    print("min:", min(eigval_com))
    print("max:", max(eigval_com))
    eigval_sim[eigval_sim <=0] = 1e-8
    eigval_sim[eigval_sim >=2]=1.999999
    eigval_com[eigval_com <=0] = 1e-8
    eigval_com[eigval_com >=2]=1.999999
    plt.figure()
    #for i in range(features.shape[1]):
    for i in range(5):
        x = features[:, i]
        #score_high = x.T.dot(L.dot(x))
        all = x.T.dot(x)

        # 计算频谱
        spectral = np.dot(eigvec_sim.T, x)

        x_f = spectral * spectral /all
        # 以特征值为横坐标画出频谱图
        plt.cla()
        #plt.plot(eigval_sim[eigval_sim < 2], x_f[eigval_sim < 2] , label='sim')
        plt.plot(eigval_sim, x_f, label='sim')
        bins = np.arange(0, 2.2, 0.2)
        digitized = np.digitize(eigval_sim, bins, right=True)
        sim_bin_means = [x_f[digitized == i].sum() for i in range(1, len(bins))]
        sim_bin_means_sum = sim_bin_means.copy()
        for j in range(1, len(sim_bin_means_sum)):
            sim_bin_means_sum[j]=sim_bin_means_sum[j-1]+sim_bin_means_sum[j]

        spectral = np.dot(eigvec_com.T, x)
        x_f = spectral * spectral /all
        '''
        score_high = 0
        for i in range(len(spectral)):
            score_high += spectral[i] * spectral[i] * eigval_com[i]
        score_high /= all
        print("score high:",score_high)
        '''
        #plt.plot(eigval_com[eigval_com < 2], x_f[eigval_com < 2], label='com')
        '''
        plt.plot(eigval_com, x_f, label='com')
        plt.xlabel('λ')
        plt.ylabel('Spectral Energy')
        plt.legend()
        plt.title('Feature Dim '+ str(i-1000))
        plt.show()
        plt.savefig(save_path + '/dim_'+ str(i) + '_曲线图.pdf', bbox_inches='tight')
        '''

        # 将曲线按照区间划分，统计每个区间内的纵坐标和
        digitized = np.digitize(eigval_com, bins,right=True)
        com_bin_means = [x_f[digitized == i].sum() for i in range(1, len(bins))]
        com_bin_means_sum = com_bin_means.copy()
        for j in range(1, len(com_bin_means_sum)):
            com_bin_means_sum[j]=com_bin_means_sum[j-1]+com_bin_means_sum[j]

        # 生成柱状图
        '''
        if com_bin_means[4]+com_bin_means[5]+com_bin_means[6]<0.2:
            continue
        else:
            print("dim ",i,"满足")
        '''
        plt.cla()
        print(bins)
        print(com_bin_means)
        print(com_bin_means_sum)
        #plt.bar(bins[:-1]+0.1, sim_bin_means, width=0.15, label='Substitute')
        plt.bar(bins[:-1]+0.1, com_bin_means, width=0.15)
        #plt.plot(bins[:-1], sim_bin_means, label='Substitute')
        #plt.plot(bins[:-1], com_bin_means, label='Complement')
        plt.plot(bins[:-1]+0.1, com_bin_means_sum)
        plt.xticks()
        plt.xlabel('λ',fontsize = 30)
        plt.ylabel('Spectral Energy', fontsize=30)
        plt.legend()
        plt.title('Feature Dim ' + str(i-2))
        plt.xticks(bins, fontsize = 10)
        plt.show()
        plt.savefig(save_path + '/dim_' + str(i)  + '_bin图.pdf', bbox_inches='tight')

        plt.cla()
        #plt.bar(bins[:-1], sim_bin_means, width=0.2, label='sim')
        #plt.bar(bins[:-1], com_bin_means, width=0.2, label='com')
        #plt.plot(bins[:-1], sim_bin_means_sum, label='Substitute')
        plt.plot(bins[:-1], com_bin_means_sum, label='Complement')
        plt.xlabel('λ')
        plt.ylabel('Spectral Energy')
        plt.legend()
        plt.title('Dim_' + str(i))
        #plt.show()
        #plt.savefig(save_path + '/dim_' + str(i) + '_累计bin图.pdf', bbox_inches='tight')

parser = argparse.ArgumentParser(description='GCO_analyze')
parser.add_argument('--dataset', type=str, default='Appliances')

args = parser.parse_args()

if __name__ == '__main__':

    print('Dataset: {}'.format(args.dataset))
    dataset = args.dataset
    sim_edge_index = []
    com_edge_index = []
    #features = []
    dataset_path = './data_preprocess2/data/' + str(dataset) + '.json'
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
    '''
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

    features = np.concatenate((cid2_emb_feature, cid3_emb_feature, np.expand_dims(features[:,2], 1)), axis=1)

    features_emb = np.squeeze(np.array(features))
    '''

    path = './data_preprocess2/embs/' + str(dataset) + '_img_embeddings.npz'
    data = np.load(path, allow_pickle=True)
    features_emb = data['img_emb']
    print(features_emb.shape)


    sim_edge_index = list(set(tuple(i) for i in sim_edge_index) - set(tuple(i) for i in com_edge_index))

    sim_edge_index = list(set(tuple(i) for i in sim_edge_index) - set(tuple(i) for i in com_edge_index))


    L_sim = generate_adj_normalize(sim_edge_index, len(features_emb))

    L_com = generate_adj_normalize(com_edge_index, len(features_emb))

    plot_frequency(features_emb, L_sim, L_com, dataset, save_path = './frequency_analysis')
