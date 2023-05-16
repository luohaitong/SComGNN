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

class Item_GraphConvolution(nn.Module):

    def __init__(self, features_size, embedding_size, bias=True):

        super(Item_GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj):

        support = torch.relu(torch.mm(feature, self.weight))
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class Item_GraphConvolution_mid(nn.Module):

    def __init__(self, features_size, embedding_size, bias=True):

        super(Item_GraphConvolution_mid, self).__init__()
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.nn_cat = nn.Linear(2 * embedding_size, embedding_size)
    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj):

        support = torch.relu(torch.mm(feature, self.weight))

        indices = torch.cat((torch.arange(adj.shape[0]).unsqueeze(0), torch.arange(adj.shape[0]).unsqueeze(0)), dim=0)
        values = torch.ones(adj.shape[0])
        eye_matrix = torch.sparse_coo_tensor(indices, values, torch.Size([adj.shape[0], adj.shape[0]])).to(feature.device)

        adj_low = adj + eye_matrix
        output_low = torch.spmm(adj_low, support)

        adj_mid = torch.sparse.mm(adj, adj) - eye_matrix
        output_mid = torch.spmm(adj_mid, support)

        #adj_high = eye_matrix - adj
        #output_high = torch.spmm(adj_high, support)

        output = output_low
        #output = self.nn_cat(torch.cat([output_low, output_mid], dim=1))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class Item_GraphConvolution_mid_attention(nn.Module):

    def __init__(self, features_size, embedding_size, bias=True, alpha=0.2):

        super(Item_GraphConvolution_mid_attention, self).__init__()
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.nn_attention = nn.Linear(2 * embedding_size, embedding_size)
        self.nn_cat = nn.Linear(2 * embedding_size, embedding_size)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj):

        support = torch.relu(torch.mm(feature, self.weight))

        indices = torch.cat((torch.arange(adj.shape[0]).unsqueeze(0), torch.arange(adj.shape[0]).unsqueeze(0)), dim=0)
        values = torch.ones(adj.shape[0])
        eye_matrix = torch.sparse_coo_tensor(indices, values, torch.Size([adj.shape[0], adj.shape[0]])).to(feature.device)

        adj_low = adj + eye_matrix
        output_low = torch.spmm(adj_low, support)

        adj_mid = torch.sparse.mm(adj, adj) - eye_matrix
        output_mid = torch.spmm(adj_mid, support)

        #adj_high = eye_matrix - adj
        #output_high = torch.spmm(adj_high, support)
        '''
        a = torch.cat([output_low, output_mid], dim=1)

        a = a.unsqueeze(1).repeat(1, 2, 1)

        attention = self.nn_attention(a)

        x = torch.cat([output_low.unsqueeze(1), output_mid.unsqueeze(1)], dim=1)
        output = torch.sum(torch.mul(attention, x), dim=1).squeeze(1)
        '''
        output = self.leakyrelu(self.nn_cat(torch.cat([output_low, output_mid], dim=1)))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class Item_GraphConvolution_gate(nn.Module):

    def __init__(self, features_size, embedding_size, bias=True):

        super(Item_GraphConvolution_gate, self).__init__()
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.nn_cat = nn.Linear(2 * embedding_size, embedding_size)
    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj):

        support = torch.relu(torch.mm(feature, self.weight))

        indices = torch.cat((torch.arange(adj.shape[0]).unsqueeze(0), torch.arange(adj.shape[0]).unsqueeze(0)), dim=0)
        values = torch.ones(adj.shape[0])
        eye_matrix = torch.sparse_coo_tensor(indices, values, torch.Size([adj.shape[0], adj.shape[0]])).to(feature.device)

        adj_low = adj + eye_matrix
        output_low = torch.spmm(adj_low, support)

        adj_mid = torch.sparse.mm(adj, adj) - eye_matrix
        output_mid = torch.spmm(adj_mid, support)

        adj_high = eye_matrix - adj
        output_high = torch.spmm(adj_high, support)

        output_1 = torch.mul(output_high, torch.relu(output_low + output_mid))
        output_2 = torch.mul(output_mid, torch.relu(output_low + output_high))
        output_3 = torch.mul(output_low, torch.relu(output_high + output_mid))

        output = torch.relu(output_1 + output_2 + output_3)
        #output = output_low
        #output = self.nn_cat(torch.cat([output_low, output_mid], dim=1))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class READ(nn.Module):

    def __init__(self, feature_size, embedding_size, num_layer, dropout=0.2):

        super(READ, self).__init__()
        self.dropout = dropout
        self.num_layer = num_layer
        self.nn_emb = nn.Linear(feature_size, embedding_size, bias=True)

        self.item_gc1 = Item_GraphConvolution_gate(embedding_size, embedding_size)
        self.item_gc2 = Item_GraphConvolution(embedding_size, embedding_size)
        self.item_gc3 = Item_GraphConvolution(embedding_size, embedding_size)

        self.xent_loss = True


    def forward(self, features, adj, train_set):

        item_latent = torch.relu(self.nn_emb(features))
        item_latent = self.item_gc1(item_latent, adj)
        key_emb = item_latent[train_set[:, 0]]
        pos_emb = item_latent[train_set[:, 1]]
        neg_emb = item_latent[train_set[:, 2]]

        pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(key_emb, neg_emb), dim=1)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
        #mrr = self._mrr(torch.unsqueeze(pos_scores, 1), torch.unsqueeze(neg_scores, 1))
        mrr, hr, ndcg = self.metrics_at_k(torch.unsqueeze(pos_scores, 1), torch.unsqueeze(neg_scores, 1), k=1)

        return loss, mrr, hr, ndcg

    def inference(self, features, adj, test_set):

        item_latent = torch.relu(self.nn_emb(features))
        item_latent = self.item_gc1(item_latent, adj)
        key_emb = item_latent[test_set[:, 0]]
        pos_emb = item_latent[test_set[:, 1]]
        neg_emb = item_latent[test_set[:, 2:]]

        pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
        #key_emb_repeat = key_emb.unsqueeze(dim=1).repeat(1, neg_emb.shape[1], 1)
        key_emb_repeat = key_emb.unsqueeze(dim=1)
        neg_scores = torch.sum(torch.mul(key_emb_repeat, neg_emb), dim=2)

        #mrr = self._mrr(torch.unsqueeze(pos_scores, 1), neg_scores)

        mrr, hr, ndcg = self.metrics_at_k_2(torch.unsqueeze(pos_scores, 1), neg_scores, k=10)

        return mrr, hr, ndcg


    def metrics_at_k(self, pos_scores, neg_scores, k):
        # 合并正样本和负样本的得分
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        # 获取每个样本的正样本得分和排名
        target_scores = pos_scores.squeeze(1)
        _, target_ranks = torch.sort(scores, dim=1, descending=True)
        target_ranks = torch.where(target_ranks == 0)[1]
        target_ranks[target_ranks==0] = 1e9
        # 获取前 K 个排名
        _, top_indices = torch.topk(scores, k=k, dim=1)

        # 计算 MRR@K
        mrr = torch.mean(1.0 / target_ranks.float())

        # 计算 HR@K
        hr = torch.mean((top_indices == 0).any(dim=1).float())

        if 1:
            # 计算 NDCG@K
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1).to(scores.device)
            ranking = torch.argsort(scores, dim=1, descending=True)
            ideal_ranking = torch.argsort(labels, dim=1, descending=True)
            ranking = ranking[:, :k]
            ideal_ranking = ideal_ranking[:, :k]


            # Compute the discounted cumulative gain (DCG)
            ranked_scores = torch.gather(labels, 1, ranking)
            discounts = torch.log2(torch.arange(2, ranked_scores.shape[1] + 2, device=scores.device))
            dcg = torch.sum((2 ** ranked_scores + 1) / discounts, dim=1)

            # Compute the ideal DCG (IDCG)
            ideal_scores = torch.gather(labels, 1, ideal_ranking)
            ideal_dcg = torch.sum((2 ** ideal_scores + 1) / discounts, dim=1)

            # Compute the NDCG
            ndcg = dcg / ideal_dcg
            ndcg = torch.mean(ndcg)

        return mrr, hr, ndcg

    def metrics_at_k_2(self, pos_scores, neg_scores, k):
        # 合并正样本和负样本的得分
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1).to(scores.device)

        ranking = torch.argsort(scores, dim=1, descending=True)
        ideal_ranking = torch.argsort(labels, dim=1, descending=True)
        ranking_k = ranking[:, :k]
        ideal_ranking_k = ideal_ranking[:, :k]

        match_score = torch.gather(labels, 1, ranking)
        index_matrix = torch.arange(1, match_score.shape[1] + 1, device=scores.device).unsqueeze(0).repeat(
            match_score.shape[0], 1)

        match_score = torch.mul(match_score, index_matrix)
        match_score[match_score == 0] = 1e9
        mrr = torch.mean(1.0 / match_score.float())
        hr = torch.mean(torch.sum(torch.gather(labels, 1, ranking_k), dim=1))

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

        return mrr, hr, ndcg

    def _mrr(self, aff, aff_neg):

        aff_all = torch.cat([aff_neg, aff], dim=1)
        size = aff_all.size(1)
        _, indices_of_ranks = torch.topk(aff_all, k=size, dim=1)
        _, ranks = torch.topk(-indices_of_ranks, k=size, dim=1)
        return torch.mean(torch.reciprocal(ranks.float()[:, -1] + 1))

    def decoder(self, tri_emb, pos_emb, neg_emb):
        logits = torch.matmul(tri_emb, pos_emb.transpose(0, 1))
        neg_logits = torch.matmul(tri_emb, neg_emb.transpose(0, 1))

        if self.xent_loss:
            true_xent = torch.nn.functional.binary_cross_entropy_with_logits(
                input=logits, target=torch.ones_like(logits), reduction='sum')
            negative_xent = torch.nn.functional.binary_cross_entropy_with_logits(
                input=neg_logits, target=torch.zeros_like(neg_logits), reduction='sum')
            loss = true_xent + negative_xent
        else:
            neg_cost = torch.logsumexp(neg_logits, dim=2, keepdim=True)
            loss = -torch.sum(logits - neg_cost)

        return loss



