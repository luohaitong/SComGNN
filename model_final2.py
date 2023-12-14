import math
import time

import numpy
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import itertools
from torch.nn import init
import random
from sklearn.metrics import ndcg_score

class FusionModel(nn.Module):
    def __init__(
        self,
        input_emb_dim: int,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_emb_dim = input_emb_dim
        self.layers = nn.Sequential(
            nn.Linear(in_features=self.input_emb_dim * 2, out_features=16),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(16, out_features=1),
        )

    def forward(self, src_embs, candidate_embs, pos = True):
        #embs = torch.hstack([src_embs, candidate_embs])
        if pos:
            embs = torch.cat([src_embs, candidate_embs], dim=1)
        else:
            embs = torch.cat([src_embs, candidate_embs], dim=2)
        return self.layers(embs)

class Twostage_Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Twostage_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query1 = nn.Linear(hidden_size, hidden_size)
        self.key1 = nn.Linear(hidden_size, hidden_size)
        self.value1 = nn.Linear(hidden_size, hidden_size)
        self.query2 = nn.Linear(hidden_size, hidden_size)
        self.key2 = nn.Linear(hidden_size, hidden_size)
        self.value2 = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.nn_cat = nn.Linear(2 * hidden_size, hidden_size)


    def forward(self, other_x, x):

        batch_size = x.size(0)
        #pair-wise attention

        query = self.query1(other_x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        key = self.key1(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        value = self.value1(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]

        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attention_scores = self.softmax(attention_scores / (self.hidden_size ** 0.5))  # [batch_size, seq_len, seq_len]
        x = torch.bmm(attention_scores, value)  # [batch_size, seq_len, hidden_size]


        '''
        x1 = x[:,0,:]
        x2 = x[:,1,:]
        x1 = self.mlp1(torch.squeeze(x1))
        x2 = self.mlp2(torch.squeeze(x2))
        #x = torch.cat([x1, x2], dim=1)
        x = torch.cat([torch.unsqueeze(x1, dim=1), torch.unsqueeze(x2, dim=1)], dim=1)
        '''
        #self attention

        query = self.query2(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        key = self.key2(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        value = self.value2(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]

        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attention_scores = self.softmax(attention_scores / (self.hidden_size ** 0.5))  # [batch_size, seq_len, seq_len]
        x = torch.bmm(attention_scores, value)  # [batch_size, seq_len, hidden_size]

        #x = torch.sum(x, dim=1)
        #x = self.mlp1(x)
        x1 = x[:,0,:]
        x2 = x[:,1,:]
        x = torch.cat([x1, x2], dim=1)
        x = self.nn_cat(x)

        return x

class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GIN, self).__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        #self.eps = Parameter(torch.zeros(output_dim))
        self.eps1 = 1e-9
        self.eps2 = 1e-13
        self.bn1 = nn.BatchNorm1d(input_dim)

    def forward(self, x, adj):
        out = self.mlp1(((1+self.eps1) * x + torch.spmm(adj, x)))
        #out = self.bn1(out)
        #out = F.relu(out)  # 使用ReLU激活函数
        #out = self.mlp2((1+self.eps2) * out + torch.spmm(adj, out))

        return out

class GCN_Low(nn.Module):

    def __init__(self, features_size, embedding_size, bias=False):

        super(GCN_Low, self).__init__()
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mlp1 = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size)
        )

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj):

        '''
        support = torch.mm(feature, self.weight)

        indices = torch.cat((torch.arange(adj.shape[0]).unsqueeze(0), torch.arange(adj.shape[0]).unsqueeze(0)), dim=0)
        values = torch.ones(adj.shape[0])
        eye_matrix = torch.sparse_coo_tensor(indices, values, torch.Size([adj.shape[0], adj.shape[0]])).to(feature.device)

        adj_low = adj + eye_matrix
        output = torch.spmm(adj_low, support)
        '''
        output = torch.spmm(adj, feature)
        output = 0.5 * output + 0.5 * feature
        output = torch.mm(output, self.weight)

        if self.bias is not None:
            output += self.bias
        #output = self.mlp1(output)
        return output

class GCN_Mid(nn.Module):

    def __init__(self, features_size, embedding_size, bias=False):

        super(GCN_Mid, self).__init__()
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.mlp1 = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size)
        )

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj):

        '''
        support = torch.mm(feature, self.weight)

        indices = torch.cat((torch.arange(adj.shape[0]).unsqueeze(0), torch.arange(adj.shape[0]).unsqueeze(0)), dim=0)
        values = torch.ones(adj.shape[0])
        eye_matrix = torch.sparse_coo_tensor(indices, values, torch.Size([adj.shape[0], adj.shape[0]])).to(feature.device)

        #adj_mid = torch.sparse.mm(adj, adj) - eye_matrix
        adj_mid = eye_matrix-adj
        output = torch.spmm(adj_mid, support)
        '''
        output = torch.spmm(adj, feature)
        output = torch.spmm(adj, output)
        output = 0.5 * output - 0.5 * feature
        output = torch.mm(output, self.weight)
        if self.bias is not None:
            output += self.bias
        #output = self.mlp1(output)
        return output

class SComGNN_low(nn.Module):

    def __init__(self, features_size, embedding_size):
        super(SComGNN_low, self).__init__()
        self.gcn = GCN_Mid(features_size, embedding_size)
        self.gcn2 = GCN_Low(features_size, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, feature, adj):

        embedding = self.bn(self.gcn2(feature, adj))
        '''
        embedding = torch.relu(embedding)
        embedding = self.gcn2(embedding, adj)
        '''

        return embedding

class SComGNN_concat(nn.Module):

    def __init__(self, features_size, embedding_size):
        super(SComGNN_concat, self).__init__()
        self.gcn_low = GCN_Low(features_size, embedding_size)
        self.gcn_mid = GCN_Mid(features_size, embedding_size)
        self.nn_cat = nn.Linear(2 * embedding_size, embedding_size)
        self.gcn_low2 = GCN_Low(features_size, embedding_size)
        self.gcn_mid2 = GCN_Mid(features_size, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)

    def forward(self, feature, adj):
        output_low = self.bn1(self.gcn_low(feature, adj))
        output_mid = self.bn2(self.gcn_mid(feature, adj))
        '''
        output_low = torch.relu(output_low)
        output_mid = torch.relu(output_mid)

        output_low = self.gcn_low2(output_low, adj)
        output_mid = self.gcn_mid2(output_mid, adj)
        '''

        output = (self.nn_cat(torch.cat([output_low, output_mid], dim=1)))

        return output

class SComGNN_Att(nn.Module):

    def __init__(self, features_size, embedding_size):
        super(SComGNN_Att, self).__init__()
        self.gcn_low = GCN_Low(features_size, embedding_size)
        self.gcn_mid = GCN_Mid(features_size, embedding_size)
        self.gcn_low2 = GCN_Low(features_size, embedding_size)
        self.gcn_mid2 = GCN_Mid(features_size, embedding_size)
        self.gcn_low3 = GCN_Low(features_size, embedding_size)
        self.gcn_mid3 = GCN_Mid(features_size, embedding_size)
        self.gcn_low4 = GCN_Low(features_size, embedding_size)
        self.gcn_mid4 = GCN_Mid(features_size, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.bn3 = nn.BatchNorm1d(embedding_size)
        self.bn4 = nn.BatchNorm1d(embedding_size)
        self.bn5 = nn.BatchNorm1d(embedding_size)
        self.bn6 = nn.BatchNorm1d(embedding_size)

    def forward(self, feature, adj):
        output_low = self.bn1(self.gcn_low(feature, adj))
        output_mid = self.bn2(self.gcn_mid(feature, adj))
        '''
        output_low = torch.relu(output_low)
        output_mid = torch.relu(output_mid)

        output_low = self.gcn_low2(output_low, adj)
        output_mid = self.gcn_mid2(output_mid, adj)

        output_low = torch.relu(self.bn3(output_low))
        output_mid = torch.relu(self.bn4(output_mid))

        output_low = self.gcn_low3(output_low, adj)
        output_mid = self.gcn_mid3(output_mid, adj)

        output_low = torch.relu(self.bn5(output_low))
        output_mid = torch.relu(self.bn6(output_mid))

        output_low = self.gcn_low4(output_low, adj)
        output_mid = self.gcn_mid4(output_mid, adj)
        '''

        output = torch.cat([torch.unsqueeze(output_low, dim=1), torch.unsqueeze(output_mid, dim=1)], dim=1)

        return output

class READ(nn.Module):

    def __init__(self, feature_size, embedding_size, price_n_bins, mode, dropout=0.2):

        super(READ, self).__init__()
        self.dropout = dropout
        self.embedding_cid2 = nn.Linear(768, embedding_size, bias=True)
        self.embedding_cid3 = nn.Linear(768, embedding_size, bias=True)
        self.embedding_price = nn.Embedding(price_n_bins, embedding_size)
        self.nn_emb = nn.Linear(embedding_size * 3, embedding_size)
        self.two_att = Twostage_Attention(embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.mode = mode
        if mode == 3:
            self.item_gc = SComGNN_Att(embedding_size, embedding_size)
        elif mode == 2:
            self.item_gc = SComGNN_concat(embedding_size, embedding_size)
        else:
            self.item_gc = SComGNN_low(embedding_size, embedding_size)
        self.fusion = FusionModel(embedding_size, 0.2)

    def forward(self, features, price, adj, train_set):

        cid2 = features[:,:768]
        cid3 = features[:,768:]
        # 将三个嵌入向量拼接在一起
        embedded_cid2 = self.embedding_cid2(cid2)
        embedded_cid3 = self.embedding_cid3(cid3)
        embed_price = self.embedding_price(price)
        item_latent = torch.relu(self.nn_emb(torch.cat([embedded_cid2, embedded_cid3, embed_price], dim=1)))
        item_latent = self.item_gc(item_latent, adj)

        key_emb = item_latent[train_set[:, 0]]
        pos_emb = item_latent[train_set[:, 1]]
        neg_emb = item_latent[train_set[:, 2:]]

        if self.mode == 3:
            key_latent_pos = self.two_att(pos_emb, key_emb)
            pos_latent = self.two_att(key_emb, pos_emb)
            for i in range(neg_emb.shape[1]):
                neg_emb_tmp = neg_emb[:, i, :, :]
                key_latent_neg_tmp = self.two_att(neg_emb_tmp, key_emb)
                neg_latent_tmp = self.two_att(key_emb, neg_emb_tmp)
                if i == 0:
                    key_latent_neg = key_latent_neg_tmp.unsqueeze(dim=1)
                    neg_latent = neg_latent_tmp.unsqueeze(dim=1)
                else:
                    key_latent_neg = torch.cat((key_latent_neg, key_latent_neg_tmp.unsqueeze(dim=1)), dim=1)
                    neg_latent = torch.cat((neg_latent, neg_latent_tmp.unsqueeze(dim=1)), dim=1)
                del key_latent_neg_tmp, neg_latent_tmp
                torch.cuda.empty_cache()
            pos_scores = torch.sum(torch.mul(key_latent_pos, pos_latent), dim=1)
            neg_scores = torch.sum(torch.mul(key_latent_neg, neg_latent), dim=2)
            #pos_scores = self.fusion(key_latent_pos, pos_latent)
            #neg_scores = self.fusion(key_latent_neg, neg_latent, False)
        else:

            pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
            key_emb = key_emb.unsqueeze(dim=1)
            neg_scores = torch.sum(torch.mul(key_emb, neg_emb), dim=2)

            '''
            pos_scores = self.fusion(key_emb, pos_emb)
            key_emb = key_emb.unsqueeze(dim=1).repeat(1, neg_emb.shape[1], 1)
            neg_scores = self.fusion(key_emb, neg_emb, False)
            '''

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-9))

        return loss

    def inference(self, features, price, adj, test_set):

        cid2 = features[:,:768]
        cid3 = features[:,768:]
        # 将三个嵌入向量拼接在一起
        embedded_cid2 = self.embedding_cid2(cid2)
        embedded_cid3 = self.embedding_cid3(cid3)
        embed_price = self.embedding_price(price)
        item_latent = torch.relu(self.nn_emb(torch.cat([embedded_cid2, embedded_cid3, embed_price], dim=1)))
        item_latent = self.item_gc(item_latent, adj)

        key_emb = item_latent[test_set[:, 0]]
        pos_emb = item_latent[test_set[:, 1]]
        neg_emb = item_latent[test_set[:, 2:]]

        if self.mode == 3:
            key_latent_pos = self.two_att(pos_emb, key_emb)
            pos_latent = self.two_att(key_emb, pos_emb)
            for i in range(neg_emb.shape[1]):
                neg_emb_tmp = neg_emb[:, i, :, :]
                key_latent_neg_tmp = self.two_att(neg_emb_tmp, key_emb)
                neg_latent_tmp = self.two_att(key_emb, neg_emb_tmp)
                if i == 0:
                    key_latent_neg = key_latent_neg_tmp.unsqueeze(dim=1)
                    neg_latent = neg_latent_tmp.unsqueeze(dim=1)
                else:
                    key_latent_neg = torch.cat((key_latent_neg, key_latent_neg_tmp.unsqueeze(dim=1)), dim=1)
                    neg_latent = torch.cat((neg_latent, neg_latent_tmp.unsqueeze(dim=1)), dim=1)

            pos_scores = torch.sum(torch.mul(key_latent_pos, pos_latent), dim=1)
            neg_scores = torch.sum(torch.mul(key_latent_neg, neg_latent), dim=2)
            #pos_scores = self.fusion(key_latent_pos, pos_latent)
            #neg_scores = self.fusion(key_latent_neg, neg_latent, False)
        else:

            pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
            key_emb = key_emb.unsqueeze(dim=1)
            neg_scores = torch.sum(torch.mul(key_emb, neg_emb), dim=2)

            '''
            pos_scores = self.fusion(key_emb, pos_emb)
            key_emb = key_emb.unsqueeze(dim=1).repeat(1, neg_emb.shape[1], 1)
            neg_scores = self.fusion(key_emb, neg_emb, False)
            '''

        mrr, hr5, hr10, ndcg, cov10 = self.metrics_at_k(torch.unsqueeze(pos_scores, 1), neg_scores, test_set[:,1:], k=10)

        return mrr, hr5, hr10, ndcg, cov10

    def metrics_at_k(self, pos_scores, neg_scores, test_set, k):
        # 合并正样本和负样本的得分
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1).to(scores.device)
        scores = torch.squeeze(scores)
        labels = torch.squeeze(labels)
        '''
        batch_size = scores.shape[0]  # 矩阵的行数
        score_num = scores.shape[1]  # 每行的元素个数
        shuffle_matrix = torch.zeros(batch_size, score_num).to(scores.device)
        random.seed(1)
        for i in range(batch_size):
            # 生成从 0 到 B-1 的数组
            arr = torch.arange(score_num)
            # 随机打乱数组
            random.shuffle(arr)
            # 将打乱后的数组赋值给矩阵的当前行
            shuffle_matrix[i] = arr
        labels = torch.gather(labels,  1, shuffle_matrix)
        scores = torch.gather(scores,  1, shuffle_matrix)
        '''
        ranking = torch.argsort(scores, dim=1, descending=True)
        ideal_ranking = torch.argsort(labels, dim=1, descending=True)

        #MRR
        match_score = torch.gather(labels, 1, ranking)
        index_matrix = torch.arange(1, match_score.shape[1] + 1, device=scores.device).unsqueeze(0).repeat(
            match_score.shape[0], 1)

        match_score = torch.mul(match_score, index_matrix)
        match_score[match_score == 0] = 1e9
        mrr = torch.mean(1.0 / match_score.float())

        #NDCG
        ranked_scores = torch.gather(labels, 1, ranking)
        discounts = torch.log2(torch.arange(2, ranked_scores.shape[1] + 2, device=scores.device))
        dcg = torch.sum((2 ** ranked_scores - 1) / discounts, dim=1)
        # Compute the ideal DCG (IDCG)
        ideal_scores = torch.gather(labels, 1, ideal_ranking)
        ideal_dcg = torch.sum((2 ** ideal_scores - 1) / discounts, dim=1)

        # Compute the NDCG
        ndcg = dcg / ideal_dcg
        ndcg = torch.mean(ndcg)

        k_list = [5, 10]
        hr_list = []
        for k in k_list:
            ranking_k = ranking[:, :k]
            ideal_ranking_k = ideal_ranking[:, :k]
            #HR@K
            hr = torch.mean(torch.sum(torch.gather(labels, 1, ranking_k), dim=1))
            hr_list.append(hr)
            '''
            # 计算 NDCG@K
            # Compute the discounted cumulative gain (DCG)
            ranked_scores = torch.gather(labels, 1, ranking_k)
            discounts = torch.log2(torch.arange(2, ranked_scores.shape[1] + 2, device=scores.device))
            dcg = torch.sum((2 ** ranked_scores - 1) / discounts, dim=1)
            # Compute the ideal DCG (IDCG)
            ideal_scores = torch.gather(labels, 1, ideal_ranking_k)
            ideal_dcg = torch.sum((2 ** ideal_scores - 1) / discounts, dim=1)

            # Compute the NDCG
            ndcg = dcg / ideal_dcg
            ndcg = torch.mean(ndcg)
            '''

            ndcg = ndcg_score(labels.cpu(), scores.cpu())

            recommend_idx = torch.gather(test_set, 1, ranking_k)
            '''
            if k == 10 :
                test_idx = torch.cat([test_set[:,:2], recommend_idx], dim=1)
            '''
            recommend_idx = recommend_idx.unique()
            all_idx = test_set.unique()
            cov10 = len(recommend_idx)/len(all_idx)

        return mrr, hr_list[0], hr_list[1], ndcg, cov10