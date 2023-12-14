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

        #output = output_low
        output = self.nn_cat(torch.cat([output_low, output_mid], dim=1))
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

class Item_GraphConvolution_reconstruct(nn.Module):

    def __init__(self, features_size, embedding_size, bias=True):

        super(Item_GraphConvolution_reconstruct, self).__init__()
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.nn_cat = nn.Linear(2 * embedding_size, embedding_size)
        self.criterion = nn.BCELoss()

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

        #output = output_low
        output = self.nn_cat(torch.cat([output_low, output_mid], dim=1))

        adj_low = torch.mm(output_low, torch.t(output_low))
        adj_low = torch.sigmoid(adj_low)
        adj_mid = torch.mm(output_mid, torch.t(output_mid))
        adj_mid = torch.sigmoid(adj_mid)

        adj_reconstruct = adj_low + adj_mid
        adj_reconstruct[adj_reconstruct>1] =1

        loss = self.criterion(adj_reconstruct.reshape(-1,1), adj.to_dense().reshape(-1,1).float())

        if self.bias is not None:
            return output + self.bias, loss
        else:
            return output, loss

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        query = self.query(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        key = self.key(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        value = self.value(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]

        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attention_scores = self.softmax(attention_scores / (self.hidden_size ** 0.5))  # [batch_size, seq_len, seq_len]
        context = torch.bmm(attention_scores, value)  # [batch_size, seq_len, hidden_size]
        context = torch.sum(context, dim=1)
        return context

class PairWiseAttention2(nn.Module):
    def __init__(self, hidden_size):
        super(PairWiseAttention2, self).__init__()
        self.hidden_size = hidden_size
        self.query1 = nn.Linear(hidden_size, hidden_size)
        self.key1 = nn.Linear(hidden_size, hidden_size)
        self.value1 = nn.Linear(hidden_size, hidden_size)
        self.query2 = nn.Linear(hidden_size, hidden_size)
        self.key2 = nn.Linear(hidden_size, hidden_size)
        self.value2 = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, other_x, x):

        batch_size = x.size(0)
        #pair-wise attention
        query = self.query1(other_x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        key = self.key1(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        value = self.value1(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]

        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attention_scores = self.softmax(attention_scores / (self.hidden_size ** 0.5))  # [batch_size, seq_len, seq_len]
        x = torch.bmm(attention_scores, value)  # [batch_size, seq_len, hidden_size]

        #self attention
        query = self.query2(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        key = self.key2(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        value = self.value2(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]

        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attention_scores = self.softmax(attention_scores / (self.hidden_size ** 0.5))  # [batch_size, seq_len, seq_len]
        x = torch.bmm(attention_scores, value)  # [batch_size, seq_len, hidden_size]

        x = torch.sum(x, dim=1)

        return x

class PairWiseAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W1 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input1, input2):
        # 计算注意力权重
        score = torch.matmul(self.W1(input1), self.W2(input2).transpose(-2, -1))
        weight1 = self.softmax(score)
        weight2 = self.softmax(score.transpose(-2, -1))

        # 计算加权和
        output1 = torch.matmul(weight1, input2)
        output2 = torch.matmul(weight2, input1)

        return output1, output2

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=2)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size = x.size(0)
        query = self.query(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_size]
        key = self.key(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_size]
        value = self.value(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_size]

        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = attention_scores / (self.head_size ** 0.5)
        attention_weights = self.softmax(attention_scores)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, value)  # [batch_size, num_heads, seq_len, head_size]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        output = self.output(context)
        output = torch.sum(output, dim=1)

        return output

class ContrastiveLoss(nn.Module):
    def __init__(self, T=0.2):
        super(ContrastiveLoss, self).__init__()
        self.T = T
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output_low, output_mid):

        batch_size = output_low.shape[0]

        logits = torch.matmul(output_low, output_mid.T)  # 计算两两样本间点乘相似度

        mask = ~torch.eye(batch_size, dtype=bool).to(output_low.device)
        pos_logits = torch.diag(logits).unsqueeze(1)
        neg_logits = torch.masked_select(logits, mask).reshape(batch_size, batch_size - 1)
        #neg_logits = logits

        logits_final = torch.cat([pos_logits, neg_logits], dim=1)

        logits_final /= self.T

        labels = torch.zeros(logits_final.shape[0], dtype=torch.long).to(output_low.device)

        loss = self.criterion(logits_final, labels)

        return loss

class ContrastiveLoss2(nn.Module):
    def __init__(self, T=0.5):
        super(ContrastiveLoss2, self).__init__()
        self.T = T
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output_low, output_mid):

        logits = torch.matmul(output_low, output_mid.T)  # 计算两两样本间点乘相似度

        #mask = torch.eye(batch_size, dtype=bool).to(output_low.device)
        logits = torch.log(torch.softmax(logits, dim=0)+ 1e-9)
        pos_logits = torch.diag(logits)
        #neg_logits = torch.masked_select(logits, mask).reshape(batch_size, batch_size - 1)
        #neg_logits = logits

        #logits_final = torch.cat([pos_logits, neg_logits], dim=1)

        #logits_final /= self.T

        #labels = torch.zeros(logits_final.shape[0], dtype=torch.long).to(output_low.device)

        #loss = self.criterion(logits_final, labels)
        loss = -torch.mean(pos_logits,dim=0)
        return loss

class ContrastiveLoss3(nn.Module):
    def __init__(self, T=0.5):
        super(ContrastiveLoss3, self).__init__()
        self.T = T
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output_low, output_mid):

        logits = torch.matmul(output_low, output_mid.T)  # 计算两两样本间点乘相似度
        logits_low = torch.matmul(output_low, output_low.T)
        logits_mid = torch.matmul(output_mid, output_mid.T)

        batch_size = output_low.shape[0]
        mask = ~torch.eye(batch_size, dtype=bool).to(output_low.device)
        pos_logits = torch.diag(logits).unsqueeze(1)
        neg_logits_low = torch.masked_select(logits_low, mask).reshape(batch_size, batch_size - 1)
        neg_max_logits_low, _ = torch.max(neg_logits_low, dim=1, keepdim=True)
        neg_logits_mid = torch.masked_select(logits_mid, mask).reshape(batch_size, batch_size - 1)
        neg_max_logits_mid, _ = torch.max(neg_logits_mid, dim=1, keepdim=True)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_logits - (neg_max_logits_low + neg_max_logits_mid)*0.5) + 1e-9))
        #logits_final /= self.T

        return loss

class Graph_Autoenconder(nn.Module):
    def __init__(self):
        super(Graph_Autoenconder, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, output_low, output_mid, adj):

        adj = adj.to_dense()
        adj_low = torch.matmul(output_low, output_low.t())
        adj_mid = torch.matmul(output_mid, output_mid.t())
        adj_res = torch.sigmoid((adj_low + adj_mid)/2)

        '''
        row_min, _ = torch.min(adj_res, dim=1, keepdim=True)
        row_max, _ = torch.max(adj_res, dim=1, keepdim=True)

        adj_res = (adj_res - row_min) / (row_max - row_min)
        '''
        adj_res = adj_res.view(-1)
        adj = adj.view(-1)

        neg_indices = torch.where(adj == 0)[0]
        indices = torch.randperm(len(neg_indices))[:int(len(neg_indices)/4)]
        sampled_neg_indices = neg_indices[indices]
        pos_indices = torch.where(adj!=0)[0]

        samepled_adj = torch.cat([adj_res[pos_indices], adj_res[sampled_neg_indices]], dim=0)
        samepled_label = torch.cat([torch.ones_like(adj_res[pos_indices]), torch.zeros_like(adj_res[sampled_neg_indices])], dim=0)

        loss = self.criterion(samepled_adj, samepled_label)

        return loss

class Item_GraphConvolution_mid_attention(nn.Module):

    def __init__(self, features_size, embedding_size, bias=False):

        super(Item_GraphConvolution_mid_attention, self).__init__()
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.nn_cat = nn.Linear(2 * embedding_size, embedding_size)
        self.self_att = SelfAttention(embedding_size)
        self.self_att2 = SelfAttention(embedding_size)
        self.parwise_att = PairWiseAttention(embedding_size)
        self.gad = Graph_Autoenconder()

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

        #output_low, output_mid = self.parwise_att(output_low, output_mid)
        #adj_high = eye_matrix - adj
        #output_high = torch.spmm(adj_high, support)

        output = torch.cat([torch.unsqueeze(output_low, dim=1), torch.unsqueeze(output_mid, dim=1)], dim=1)
        #output_key = self.self_att(output)
        #output_com = self.self_att2(output)

        #loss_gad = self.gad(output_low, output_mid, adj)

        if self.bias is not None:
            return output + self.bias, output_low, output_mid
        else:
            return output, output_low, output_mid

class Item_GraphConvolution_mid2(nn.Module):

    def __init__(self, features_size, embedding_size, bias=False):

        super(Item_GraphConvolution_mid2, self).__init__()
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(embedding_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.nn_cat = nn.Linear(2 * embedding_size, embedding_size)
        self.self_att = SelfAttention(embedding_size)
        self.self_att2 = SelfAttention(embedding_size)
        self.parwise_att = PairWiseAttention(embedding_size)
        self.gad = Graph_Autoenconder()

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

        #output_low, output_mid = self.parwise_att(output_low, output_mid)
        #adj_high = eye_matrix - adj
        #output_high = torch.spmm(adj_high, support)

        output = torch.cat([torch.unsqueeze(output_low, dim=1), torch.unsqueeze(output_mid, dim=1)], dim=1)
        #output_key = self.self_att(output)
        #output_com = self.self_att2(output)

        #loss_gad = self.gad(output_low, output_mid, adj)

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

        self.item_gc1 = Item_GraphConvolution_mid_attention(embedding_size, embedding_size)
        self.item_gc2 = Item_GraphConvolution(embedding_size, embedding_size)
        self.item_gc3 = Item_GraphConvolution(embedding_size, embedding_size)

        self.xent_loss = True


    def forward(self, features, adj, train_set, epoch):

        item_latent = torch.relu(self.nn_emb(features))
        item_latent_key, item_latent_com, _, _ = self.item_gc1(item_latent, adj)
        key_emb = item_latent_key[train_set[:, 0]]
        pos_emb = item_latent_key[train_set[:, 1]]
        neg_emb = item_latent_key[train_set[:, 2]]

        pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(key_emb, neg_emb), dim=1)


        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
        #mrr = self._mrr(torch.unsqueeze(pos_scores, 1), torch.unsqueeze(neg_scores, 1))
        mrr, hr, ndcg = self.metrics_at_k(torch.unsqueeze(pos_scores, 1), torch.unsqueeze(neg_scores, 1), k=1)

        return loss, mrr, hr, ndcg

    def inference(self, features, adj, test_set):

        item_latent = torch.relu(self.nn_emb(features))
        item_latent_key, item_latent_com, item_latent_low, item_latent_mid = self.item_gc1(item_latent, adj)
        key_emb = item_latent_key[test_set[:, 0]]
        pos_emb = item_latent_key[test_set[:, 1]]
        neg_emb = item_latent_key[test_set[:, 2:]]

        pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
        #key_emb_repeat = key_emb.unsqueeze(dim=1).repeat(1, neg_emb.shape[1], 1)
        key_emb_repeat = key_emb.unsqueeze(dim=1)
        neg_scores = torch.sum(torch.mul(key_emb_repeat, neg_emb), dim=2)

        #mrr = self._mrr(torch.unsqueeze(pos_scores, 1), neg_scores)

        mrr, hr, ndcg, scores = self.metrics_at_k_2(torch.unsqueeze(pos_scores, 1), neg_scores, k=10)

        adj_low = torch.sigmoid(torch.matmul(item_latent_low, item_latent_low.t()))
        adj_mid = torch.sigmoid(torch.matmul(item_latent_mid, item_latent_mid.t()))

        return mrr, hr, ndcg, adj_low, adj_mid, scores

    def row_normalize(self, adj):

        row_sum = torch.sum(adj, dim=1)
        adj_normalized = adj / row_sum.unsqueeze(1)
        adj_normalized[torch.isinf(adj_normalized)] = 0

        return adj_normalized

    def row_minmax(self, adj):

        row_min, _ = torch.min(adj, dim=1, keepdim=True)
        row_max, _ = torch.max(adj, dim=1, keepdim=True)
        adj_minmax = (adj - row_min) / (row_max - row_min)

        return adj_minmax

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

        return mrr, hr, ndcg, scores

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

class READ_as(nn.Module):

    def __init__(self, feature_size, embedding_size, num_layer, dropout=0.2):

        super(READ_as, self).__init__()
        self.dropout = dropout
        self.num_layer = num_layer
        self.nn_emb = nn.Linear(feature_size, embedding_size, bias=True)

        self.item_gc_key = Item_GraphConvolution_mid_attention(embedding_size, embedding_size)
        self.item_gc_com = Item_GraphConvolution_mid_attention(embedding_size, embedding_size)

        self.xent_loss = True


    def forward(self, features, adj, train_set):

        item_latent = torch.relu(self.nn_emb(features))
        item_latent_key = self.item_gc_key(item_latent, adj)
        item_latent_com = self.item_gc_com(item_latent, adj)
        key_emb = item_latent_key[train_set[:, 0]]
        pos_emb = item_latent_com[train_set[:, 1]]
        neg_emb = item_latent_com[train_set[:, 2]]

        pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(key_emb, neg_emb), dim=1)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
        #mrr = self._mrr(torch.unsqueeze(pos_scores, 1), torch.unsqueeze(neg_scores, 1))
        mrr, hr, ndcg = self.metrics_at_k(torch.unsqueeze(pos_scores, 1), torch.unsqueeze(neg_scores, 1), k=1)

        return loss, mrr, hr, ndcg

    def inference(self, features, adj, test_set):

        item_latent = torch.relu(self.nn_emb(features))
        item_latent_key = self.item_gc_key(item_latent, adj)
        item_latent_com = self.item_gc_com(item_latent, adj)
        key_emb = item_latent_key[test_set[:, 0]]
        pos_emb = item_latent_com[test_set[:, 1]]
        neg_emb = item_latent_com[test_set[:, 2:]]

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

class READ_pair_att(nn.Module):

    def __init__(self, feature_size, embedding_size, num_layer, dropout=0.2):

        super(READ_pair_att, self).__init__()
        self.dropout = dropout
        self.num_layer = num_layer
        self.nn_emb = nn.Linear(feature_size, embedding_size, bias=True)

        self.item_gc1 = Item_GraphConvolution_mid2(embedding_size, embedding_size)
        self.item_gc2 = Item_GraphConvolution(embedding_size, embedding_size)
        self.item_gc3 = Item_GraphConvolution(embedding_size, embedding_size)
        self.pairwise_att = PairWiseAttention2(embedding_size)
        self.self_att = SelfAttention(embedding_size)
        self.weight = nn.Parameter(torch.empty(2, embedding_size))
        nn.init.xavier_normal(self.weight)
        self.xent_loss = True
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, adj, train_set, epoch):

        item_latent = torch.relu(self.nn_emb(features))
        item_latent  = self.item_gc1(item_latent, adj)
        key_emb = item_latent[train_set[:, 0]]
        pos_emb = item_latent[train_set[:, 1]]
        neg_emb = item_latent[train_set[:, 2]]

        key_latent_pos = self.pairwise_att(pos_emb, key_emb)
        key_latent_neg = self.pairwise_att(neg_emb, key_emb)
        #key_latent = self.self_att(key_emb)
        #key_latent = key_emb
        pos_latent = self.pairwise_att(key_emb, pos_emb)
        neg_latent = self.pairwise_att(key_emb, neg_emb)

        #pos_scores = torch.sum(self.self_att(torch.mul(key_latent_pos, pos_latent)), dim=1)
        #neg_scores = torch.sum(self.self_att(torch.mul(key_latent_neg, neg_latent)), dim=1)
        #pos_scores = torch.sum(torch.sum(torch.mul((torch.mul(key_latent, pos_latent)), self.softmax(self.weight)),dim=1), dim=1)
        #neg_scores = torch.sum(torch.sum(torch.mul((torch.mul(key_latent, neg_latent)), self.softmax(self.weight)),dim=1), dim=1)
        pos_scores = torch.sum(torch.mul(key_latent_pos, pos_latent), dim=1)
        neg_scores = torch.sum(torch.mul(key_latent_pos, neg_latent), dim=1)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
        # mrr = self._mrr(torch.unsqueeze(pos_scores, 1), torch.unsqueeze(neg_scores, 1))
        mrr, hr, ndcg = self.metrics_at_k(torch.unsqueeze(pos_scores, 1), torch.unsqueeze(neg_scores, 1), k=1)

        return loss, mrr, hr, ndcg

    def inference(self, features, adj, test_set):

        item_latent = torch.relu(self.nn_emb(features))
        item_latent = self.item_gc1(item_latent, adj)
        key_emb = item_latent[test_set[:, 0]]
        pos_emb = item_latent[test_set[:, 1]]
        neg_emb = item_latent[test_set[:, 2:]]

        key_latent_pos = self.pairwise_att(pos_emb, key_emb)
        #key_latent_pos = key_emb
        pos_latent = self.pairwise_att(key_emb, pos_emb)
        for i in range(neg_emb.shape[1]):
            neg_emb_tmp = neg_emb[:,i,:,:]
            key_latent_neg_tmp = self.pairwise_att(neg_emb_tmp, key_emb)
            #key_latent_neg_tmp = key_emb
            neg_latent_tmp = self.pairwise_att(key_emb, neg_emb_tmp)
            #neg_scores_tmp = torch.sum(self.self_att(torch.mul(key_latent_neg_tmp, neg_latent_tmp)), dim=1)
            #neg_scores_tmp = torch.sum(torch.sum(torch.mul((torch.mul(key_latent_neg_tmp, neg_latent_tmp)), self.softmax(self.weight)),dim=1), dim=1)
            if i==0:
                key_latent_neg = key_latent_neg_tmp.unsqueeze(dim=1)
                neg_latent = neg_latent_tmp.unsqueeze(dim=1)
                #neg_scores = neg_scores_tmp.unsqueeze(1)
            else:
                key_latent_neg = torch.cat((key_latent_neg, key_latent_neg_tmp.unsqueeze(dim=1)), dim=1)
                neg_latent = torch.cat((neg_latent, neg_latent_tmp.unsqueeze(dim=1)), dim=1)
                #neg_scores = torch.cat((neg_scores, neg_scores_tmp.unsqueeze(dim=1)), dim=1)

        #pos_scores = torch.sum(torch.sum(torch.mul((torch.mul(key_latent_pos, pos_latent)), self.softmax(self.weight)),dim=1), dim=1)
        #pos_scores = torch.sum(self.self_att(torch.mul(key_latent_pos, pos_latent)), dim=1)

        #neg_scores = torch.sum(self.self_att(torch.mul(key_latent_neg, neg_latent)), dim=1)


        pos_scores = torch.sum(torch.mul(key_latent_pos, pos_latent), dim=1)
        # key_emb_repeat = key_emb.unsqueeze(dim=1).repeat(1, neg_emb.shape[1], 1)
        #key_latent_repeat = key_latent_neg.unsqueeze(dim=1)
        neg_scores = torch.sum(torch.mul(key_latent_neg, neg_latent), dim=2)


        # mrr = self._mrr(torch.unsqueeze(pos_scores, 1), neg_scores)

        mrr, hr, ndcg, scores = self.metrics_at_k_2(torch.unsqueeze(pos_scores, 1), neg_scores, k=10)

        #adj_low = torch.sigmoid(torch.matmul(item_latent_low, item_latent_low.t()))
        #adj_mid = torch.sigmoid(torch.matmul(item_latent_mid, item_latent_mid.t()))

        return mrr, hr, ndcg

    def row_normalize(self, adj):

        row_sum = torch.sum(adj, dim=1)
        adj_normalized = adj / row_sum.unsqueeze(1)
        adj_normalized[torch.isinf(adj_normalized)] = 0

        return adj_normalized

    def row_minmax(self, adj):

        row_min, _ = torch.min(adj, dim=1, keepdim=True)
        row_max, _ = torch.max(adj, dim=1, keepdim=True)
        adj_minmax = (adj - row_min) / (row_max - row_min)

        return adj_minmax

    def metrics_at_k(self, pos_scores, neg_scores, k):
        # 合并正样本和负样本的得分
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        # 获取每个样本的正样本得分和排名
        target_scores = pos_scores.squeeze(1)
        _, target_ranks = torch.sort(scores, dim=1, descending=True)
        target_ranks = torch.where(target_ranks == 0)[1]
        target_ranks[target_ranks == 0] = 1e9
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

        return mrr, hr, ndcg, scores

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


class READ_pair_att_npair(nn.Module):

    def __init__(self, feature_size, embedding_size, num_layer, dropout=0.2):

        super(READ_pair_att_npair, self).__init__()
        self.dropout = dropout
        self.num_layer = num_layer
        self.nn_emb = nn.Linear(feature_size, embedding_size, bias=True)

        self.item_gc1 = Item_GraphConvolution_mid(embedding_size, embedding_size)
        self.item_gc2 = Item_GraphConvolution(embedding_size, embedding_size)
        self.item_gc3 = Item_GraphConvolution(embedding_size, embedding_size)
        self.pairwise_att = PairWiseAttention2(embedding_size)
        self.self_att = SelfAttention(embedding_size)
        self.weight = nn.Parameter(torch.empty(2, embedding_size))
        nn.init.xavier_normal(self.weight)
        self.xent_loss = True
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, adj, train_set, epoch):


        item_latent = torch.relu(self.nn_emb(features))
        item_latent = self.item_gc1(item_latent, adj)
        key_emb = item_latent[train_set[:, 0]]
        pos_emb = item_latent[train_set[:, 1]]
        neg_emb = item_latent[train_set[:, 2:]]

        key_latent_pos = self.pairwise_att(pos_emb, key_emb)
        pos_latent = self.pairwise_att(key_emb, pos_emb)
        for i in range(neg_emb.shape[1]):
            neg_emb_tmp = neg_emb[:, i, :, :]
            key_latent_neg_tmp = self.pairwise_att(neg_emb_tmp, key_emb)
            # key_latent_neg_tmp = key_emb
            neg_latent_tmp = self.pairwise_att(key_emb, neg_emb_tmp)
            # neg_scores_tmp = torch.sum(self.self_att(torch.mul(key_latent_neg_tmp, neg_latent_tmp)), dim=1)
            # neg_scores_tmp = torch.sum(torch.sum(torch.mul((torch.mul(key_latent_neg_tmp, neg_latent_tmp)), self.softmax(self.weight)),dim=1), dim=1)
            if i == 0:
                key_latent_neg = key_latent_neg_tmp.unsqueeze(dim=1)
                neg_latent = neg_latent_tmp.unsqueeze(dim=1)
                # neg_scores = neg_scores_tmp.unsqueeze(1)
            else:
                key_latent_neg = torch.cat((key_latent_neg, key_latent_neg_tmp.unsqueeze(dim=1)), dim=1)
                neg_latent = torch.cat((neg_latent, neg_latent_tmp.unsqueeze(dim=1)), dim=1)

        pos_scores = torch.sum(torch.mul(key_latent_pos, pos_latent), dim=1)
        # key_emb_repeat = key_emb.unsqueeze(dim=1).repeat(1, neg_emb.shape[1], 1)
        # key_latent_repeat = key_latent_neg.unsqueeze(dim=1)
        neg_scores = torch.sum(torch.mul(key_latent_neg, neg_latent), dim=2)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-9))

        return loss, 0, 0, 0

    def inference(self, features, adj, test_set):

        item_latent = torch.relu(self.nn_emb(features))
        item_latent = self.item_gc1(item_latent, adj)
        key_emb = item_latent[test_set[:, 0]]
        pos_emb = item_latent[test_set[:, 1]]
        neg_emb = item_latent[test_set[:, 2:]]

        key_latent_pos = self.pairwise_att(pos_emb, key_emb)
        # key_latent_pos = key_emb
        pos_latent = self.pairwise_att(key_emb, pos_emb)
        for i in range(neg_emb.shape[1]):
            neg_emb_tmp = neg_emb[:, i, :, :]
            key_latent_neg_tmp = self.pairwise_att(neg_emb_tmp, key_emb)
            # key_latent_neg_tmp = key_emb
            neg_latent_tmp = self.pairwise_att(key_emb, neg_emb_tmp)
            # neg_scores_tmp = torch.sum(self.self_att(torch.mul(key_latent_neg_tmp, neg_latent_tmp)), dim=1)
            # neg_scores_tmp = torch.sum(torch.sum(torch.mul((torch.mul(key_latent_neg_tmp, neg_latent_tmp)), self.softmax(self.weight)),dim=1), dim=1)
            if i == 0:
                key_latent_neg = key_latent_neg_tmp.unsqueeze(dim=1)
                neg_latent = neg_latent_tmp.unsqueeze(dim=1)
                # neg_scores = neg_scores_tmp.unsqueeze(1)
            else:
                key_latent_neg = torch.cat((key_latent_neg, key_latent_neg_tmp.unsqueeze(dim=1)), dim=1)
                neg_latent = torch.cat((neg_latent, neg_latent_tmp.unsqueeze(dim=1)), dim=1)
                # neg_scores = torch.cat((neg_scores, neg_scores_tmp.unsqueeze(dim=1)), dim=1)

        # pos_scores = torch.sum(torch.sum(torch.mul((torch.mul(key_latent_pos, pos_latent)), self.softmax(self.weight)),dim=1), dim=1)
        # pos_scores = torch.sum(self.self_att(torch.mul(key_latent_pos, pos_latent)), dim=1)

        # neg_scores = torch.sum(self.self_att(torch.mul(key_latent_neg, neg_latent)), dim=1)

        pos_scores = torch.sum(torch.mul(key_latent_pos, pos_latent), dim=1)
        # key_emb_repeat = key_emb.unsqueeze(dim=1).repeat(1, neg_emb.shape[1], 1)
        # key_latent_repeat = key_latent_neg.unsqueeze(dim=1)
        neg_scores = torch.sum(torch.mul(key_latent_neg, neg_latent), dim=2)

        # mrr = self._mrr(torch.unsqueeze(pos_scores, 1), neg_scores)

        mrr, hr, ndcg, scores = self.metrics_at_k_2(torch.unsqueeze(pos_scores, 1), neg_scores, k=10)

        # adj_low = torch.sigmoid(torch.matmul(item_latent_low, item_latent_low.t()))
        # adj_mid = torch.sigmoid(torch.matmul(item_latent_mid, item_latent_mid.t()))

        return mrr, hr, ndcg

    def row_normalize(self, adj):

        row_sum = torch.sum(adj, dim=1)
        adj_normalized = adj / row_sum.unsqueeze(1)
        adj_normalized[torch.isinf(adj_normalized)] = 0

        return adj_normalized

    def row_minmax(self, adj):

        row_min, _ = torch.min(adj, dim=1, keepdim=True)
        row_max, _ = torch.max(adj, dim=1, keepdim=True)
        adj_minmax = (adj - row_min) / (row_max - row_min)

        return adj_minmax

    def metrics_at_k(self, pos_scores, neg_scores, k):
        # 合并正样本和负样本的得分
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        # 获取每个样本的正样本得分和排名
        target_scores = pos_scores.squeeze(1)
        _, target_ranks = torch.sort(scores, dim=1, descending=True)
        target_ranks = torch.where(target_ranks == 0)[1]
        target_ranks[target_ranks == 0] = 1e9
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

        return mrr, hr, ndcg, scores

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