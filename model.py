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

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse.FloatTensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class SpGraphAttentionLayer_batch(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer_batch, self).__init__()
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.label_clf = nn.Linear(in_dim, 2)

    def forward(self, input, adj, item_id):
        device = adj.device
        N = input.size()[0]
        edge_index = adj._indices()
        # 对边进行choose
        # edge_index = self.choost_step(input, adj, pos_edge_index)
        # 先对所有特征进行一次线性变换
        input = torch.mm(input, self.W)
        input_batch = input[item_id]
        # 连接节点与其所有邻居的表示
        hidden = torch.cat((input_batch[edge_index[0, :], :], input[edge_index[1, :], :]), dim=1).t()

        # 通过向量a得到分数
        edge_value = torch.exp(-self.leakyrelu(self.a.mm(hidden).squeeze()))  # 注意力中的分子
        e_rowsum = self.special_spmm(edge_index, edge_value, torch.Size([len(item_id), N]), torch.ones(size=(N, 1), device=device))

        # edge_value = self.dropout(edge_value)

        # edge_value = torch.mul(edge_value, torch.sign(edge_score))

        # 这也是一个技巧，乘完特征再去除，跟算完注意力再去乘特征是一个道理
        h_prime = self.special_spmm(edge_index, edge_value, torch.Size([len(item_id), N]), input)
        h_prime = h_prime.div(e_rowsum)
        self.concat = False
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class SpGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()
        self.label_clf = nn.Linear(in_dim, 2)

    def forward(self, input, adj):
        device = adj.device
        N = input.size()[0]
        edge_index = adj._indices()
        # 先对所有特征进行一次线性变换
        input = torch.mm(input, self.W)
        input_batch = input
        # 连接节点与其所有邻居的表示
        hidden = torch.cat((input_batch[edge_index[0, :], :], input[edge_index[1, :], :]), dim=1).t()

        # 通过向量a得到分数
        edge_value = torch.exp(-self.leakyrelu(self.a.mm(hidden).squeeze()))  # 注意力中的分子
        e_rowsum = self.special_spmm(edge_index, edge_value, torch.Size([N, N]), torch.ones(size=(N, 1), device=device))

        # edge_value = self.dropout(edge_value)

        # edge_value = torch.mul(edge_value, torch.sign(edge_score))

        # 这也是一个技巧，乘完特征再去除，跟算完注意力再去乘特征是一个道理
        h_prime = self.special_spmm(edge_index, edge_value, torch.Size([N, N]), input)
        #e_rowsum[e_rowsum==0] = 1e19
        h_prime = h_prime.div(e_rowsum)
        h_prime = torch.where(torch.isnan(h_prime), torch.full_like(h_prime, 1e-18), h_prime)
        self.concat = False
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.feature_size = out_features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.nn1 = nn.Linear(out_features, 1, bias=True)

    def forward(self, x, adj):
        '''
        :param x: nodes_num * embedding_size
        :param adj: nodes_num * nodes_num
        :return: nodes_num * embedding_size
        '''
        x = torch.mm(x, self.W) # shape [N, embedding_size]
        #N = x.size()[0]
        #N = adj.size()[0]

        #邻接矩阵按行归一化
        #e = torch.norm(x, dim=1)

        #e = torch.matmul(x, self.a).squeeze(1)

        #此处的attention为权重值
        #h_prime = torch.sparse.mm(attention, x)  # [batch_size,N], [N, embedding_size] --> [batch_size, embedding_size]
        batch_size = adj.size()[0]
        item_num = adj.size()[1]
        x1 = x.repeat(1, batch_size).view(batch_size * item_num, -1)
        x2 = x.repeat(batch_size, 1)
        a_input = torch.cat([x1, x2], dim=1).view(batch_size, -1, 2 * self.feature_size) # shape[N, N, 2*embedding_size]
        #e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N,N,1] -> [N,N]
        e = torch.matmul(a_input, self.a).squeeze(2)  # [N,N,1] -> [N,N]
        zero_vec = -9e15*torch.ones_like(e)
        adj = adj.to_dense()
        attention = torch.where(adj > 0, e, zero_vec)
        #此处的attention为权重值
        h_prime = torch.sparse.mm(attention, x)  # [batch_size,N], [N, embedding_size] --> [batch_size, embedding_size]

        return h_prime

class GraphAttentionLayer2(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.0):
        super(GraphAttentionLayer2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_prob = dropout_prob

        # 定义可训练的参数
        self.W = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.a = torch.nn.Parameter(torch.Tensor(2 * out_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / self.W.size(1) ** 0.5
        self.W.data.uniform_(-stdv, stdv)
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        h = torch.mm(x, self.W)
        N = h.size()[0]
        is_batch = True
        if is_batch:
            batch_size = 10
            for i in range(int(N/batch_size)+1):
                begin_idx = batch_size * i
                end_idx = min((batch_size) * (i+1), N)
                real_batch_size = end_idx-begin_idx
                h1 = h[begin_idx:end_idx,:].repeat(1, N).view(real_batch_size * N, -1)
                h2 = h[begin_idx:end_idx, :].repeat(N, 1)
                a_input = torch.cat([h1, h2], dim=1).view(real_batch_size, -1, 2 * self.out_features)
                #a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
                e_tem = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))
                del a_input, h1, h2
                torch.cuda.empty_cache()
                if i==0:
                    e = e_tem
                else:
                    e = torch.cat([e, e_tem], dim=0)
                del e_tem
                torch.cuda.empty_cache()
        else:
            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
            e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        adj = adj.to_dense()
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout_prob, training=self.training)
        h_prime = torch.matmul(attention, h)

        return F.elu(h_prime)

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

class Item_ChebConvolution(nn.Module):

    def __init__(self, features_size, embedding_size, k=3, bias=True):
        super(Item_ChebConvolution, self).__init__()
        self.k = k
        self.weight = Parameter(torch.FloatTensor(k, features_size, embedding_size))
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
        x = feature.unsqueeze(0)  # 维度转换，使得卷积能够处理
        output = torch.zeros_like(x)

        for i in range(self.k):
            if i == 0:
                T = x
            elif i == 1:
                T = adj.matmul(x)
            else:
                T = 2 * adj.matmul(T) - x_prev

            x_prev = x
            x = torch.relu(torch.bmm(T, self.weight[i]))

            output += x

        output = output.squeeze(0)  # 维度转换，使得线性层能够处理

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

class ContrastiveLoss4(nn.Module):
    def __init__(self, T=1):
        super(ContrastiveLoss4, self).__init__()
        self.T = T
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output_low, output_mid):

        logits = torch.matmul(output_low, output_mid.T)  # 计算两两样本间点乘相似度


        batch_size = output_low.shape[0]
        mask = ~torch.eye(batch_size, dtype=bool).to(output_low.device)
        pos_logits = torch.diag(logits).unsqueeze(1)
        neg_logits = torch.masked_select(logits, mask).reshape(batch_size, batch_size - 1)
        #neg_max_logits, _ = torch.max(neg_logits, dim=1, keepdim=True)
        logits_resort = torch.cat([pos_logits, neg_logits], dim=1)
        logits_resort /= self.T
        labels = torch.zeros(logits_resort.shape[0], dtype=torch.long).to(output_low.device)
        loss = self.criterion(logits_resort, labels)
        #loss = -torch.mean(torch.log(torch.sigmoid(pos_logits - neg_max_logits + 1e-9))
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
        self.weight_sim = Parameter(torch.FloatTensor(features_size, embedding_size))
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
        self.contra_loss = ContrastiveLoss4()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight_sim.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj, sim_adj=None):

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
        if sim_adj is not None:
            support_sim = torch.relu(torch.mm(feature, self.weight_sim))
            adj_low = adj + eye_matrix
            output_sim = torch.spmm(adj_low, support_sim)
            loss_con = self.contra_loss(output_mid, output_low)
        else:
            loss_con = None
        output = torch.cat([torch.unsqueeze(output_low, dim=1), torch.unsqueeze(output_mid, dim=1)], dim=1)
        output = self.self_att(output)
        #output_com = self.self_att2(output)

        #loss_gad = self.gad(output_low, output_mid, adj)

        if self.bias is not None:
            return output + self.bias, loss_con
        else:
            return output, loss_con

class Item_GraphConvolution_mid_attention_over(nn.Module):

    def __init__(self, features_size, embedding_size, bias=False):

        super(Item_GraphConvolution_mid_attention_over, self).__init__()
        self.weight = Parameter(torch.FloatTensor(features_size, embedding_size))
        self.weight_sim = Parameter(torch.FloatTensor(features_size, embedding_size))
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
        self.contra_loss = ContrastiveLoss4()

        self.num_filters = 5
        self.low = nn.ModuleList()
        self.high = nn.ModuleList()
        self.mid = nn.ModuleList()
        self.low_gamma = nn.ParameterList()
        self.mid_gamma = nn.ParameterList()
        self.high_gamma = nn.ParameterList()
        self.low_alpha = nn.ParameterList()
        self.mid_alpha = nn.ParameterList()
        self.high_alpha = nn.ParameterList()
        self.eps = 1e-9
        K = 2
        self.K = K
        for i in range(self.num_filters):
            self.low_gamma.append(torch.nn.Parameter(torch.FloatTensor([1/K for i in range(K)])))
            self.mid_gamma.append(torch.nn.Parameter(torch.FloatTensor([1/K for i in range(K)])))
            self.high_gamma.append(torch.nn.Parameter(torch.FloatTensor([1/K for i in range(K)])))
        self.alpha = torch.Tensor(np.linspace(-self.eps, 1+self.eps, self.K))
        self.midalpha = torch.Tensor(np.linspace(-self.eps, 1+self.eps, self.K))

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight_sim.data.uniform_(-stdv, stdv)



    def forward(self, feature, adj):

        support = torch.relu(torch.mm(feature, self.weight))

        indices = torch.cat((torch.arange(adj.shape[0]).unsqueeze(0), torch.arange(adj.shape[0]).unsqueeze(0)), dim=0)
        values = torch.ones(adj.shape[0])
        eye_matrix = torch.sparse_coo_tensor(indices, values, torch.Size([adj.shape[0], adj.shape[0]])).to(feature.device)

        adj_low = adj + eye_matrix
        output_low = torch.spmm(adj_low, support)

        adj_mid = torch.sparse.mm(adj, adj) - eye_matrix
        output_mid = torch.spmm(adj_mid, support)

        alpha = self.alpha.to(feature.device)
        midalpha = self.midalpha.to(feature.device)
        output_low_list = []
        output_mid_list = []
        adj_support = torch.spmm(adj, support)
        for i in range(self.num_filters):
            gamma = self.low_gamma[i]
            gamma = torch.relu(gamma)
            gamma = gamma.squeeze()
            a = torch.dot(alpha, gamma)
            b = torch.dot(1 - alpha, gamma)
            #adj_low = a * adj + b * eye_matrix
            #output_low = torch.spmm(adj_low, support)
            output_low = a * adj_support + b * support
            output_low_list.append(output_low.unsqueeze(2))

            gamma = self.mid_gamma[i]
            gamma = torch.relu(gamma)
            gamma = gamma.squeeze()
            a = torch.sum(gamma)
            c = torch.dot(midalpha, gamma)
            output_mid = a * adj_support - c * support
            #output_mid = self.mid[i](o)
            #adj_mid = a * adj - c * eye_matrix
            #output_mid = torch.spmm(adj_mid, support)
            output_mid_list.append(output_mid.unsqueeze(2))
        #output_low, output_mid = self.parwise_att(output_low, output_mid)
        #adj_high = eye_matrix - adj
        #output_high = torch.spmm(adj_high, support)
        #print(output_low_list[0].shape)
        output_low_list = torch.cat(output_low_list, dim=2)
        output_low_list = torch.sum(output_low_list, dim=2)
        output_low = output_low_list.squeeze()

        output_mid_list = torch.cat(output_mid_list, dim=2)
        output_mid_list = torch.sum(output_mid_list, dim=2)
        output_mid = output_mid_list.squeeze()
        output = torch.cat([torch.unsqueeze(output_low, dim=1), torch.unsqueeze(output_mid, dim=1)], dim=1)
        output = self.self_att(output)
        #output_com = self.self_att2(output)

        #loss_gad = self.gad(output_low, output_mid, adj)

        if self.bias is not None:
            return output + self.bias, 0
        else:
            return output, 0

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
        item_latent, loss2 = self.item_gc1(item_latent, adj)
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
        item_latent,_  = self.item_gc1(item_latent, adj)
        key_emb = item_latent[test_set[:, 0]]
        pos_emb = item_latent[test_set[:, 1]]
        neg_emb = item_latent[test_set[:, 2:]]

        pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
        #key_emb_repeat = key_emb.unsqueeze(dim=1).repeat(1, neg_emb.shape[1], 1)
        key_emb_repeat = key_emb.unsqueeze(dim=1)
        neg_scores = torch.sum(torch.mul(key_emb_repeat, neg_emb), dim=2)

        #mrr = self._mrr(torch.unsqueeze(pos_scores, 1), neg_scores)

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

class READ_assym(nn.Module):

    def __init__(self, feature_size, embedding_size, num_layer, dropout=0.2):

        super(READ_assym, self).__init__()
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

class READ_contra(nn.Module):

    def __init__(self, feature_size, embedding_size, num_layer, dropout=0.2):

        super(READ_contra, self).__init__()
        self.dropout = dropout
        self.num_layer = num_layer
        self.nn_emb = nn.Linear(feature_size, embedding_size, bias=True)

        self.item_gc1 = Item_GraphConvolution_mid_attention(embedding_size, embedding_size)
        self.item_gc2 = Item_GraphConvolution(embedding_size, embedding_size)
        self.item_gc3 = Item_GraphConvolution(embedding_size, embedding_size)

        self.xent_loss = True


    def forward(self, features, adj, train_set, epoch, sim_adj):

        item_latent = torch.relu(self.nn_emb(features))
        item_latent, loss2 = self.item_gc1(item_latent, adj, sim_adj)
        key_emb = item_latent[train_set[:, 0]]
        pos_emb = item_latent[train_set[:, 1]]
        neg_emb = item_latent[train_set[:, 2]]

        pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(key_emb, neg_emb), dim=1)


        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9)) + loss2
        #mrr = self._mrr(torch.unsqueeze(pos_scores, 1), torch.unsqueeze(neg_scores, 1))
        mrr, hr, ndcg = self.metrics_at_k(torch.unsqueeze(pos_scores, 1), torch.unsqueeze(neg_scores, 1), k=1)

        return loss, mrr, hr, ndcg

    def inference(self, features, adj, test_set):

        item_latent = torch.relu(self.nn_emb(features))
        item_latent,_  = self.item_gc1(item_latent, adj)
        key_emb = item_latent[test_set[:, 0]]
        pos_emb = item_latent[test_set[:, 1]]
        neg_emb = item_latent[test_set[:, 2:]]

        pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
        #key_emb_repeat = key_emb.unsqueeze(dim=1).repeat(1, neg_emb.shape[1], 1)
        key_emb_repeat = key_emb.unsqueeze(dim=1)
        neg_scores = torch.sum(torch.mul(key_emb_repeat, neg_emb), dim=2)

        #mrr = self._mrr(torch.unsqueeze(pos_scores, 1), neg_scores)

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

class READ_train_contra(nn.Module):

    def __init__(self, feature_size, embedding_size, num_layer, dropout=0.2, T=0.1):

        super(READ_train_contra, self).__init__()
        self.dropout = dropout
        self.num_layer = num_layer
        self.nn_emb = nn.Linear(feature_size, embedding_size, bias=True)

        self.item_gc1 = Item_GraphConvolution_mid_attention(embedding_size, embedding_size)
        self.item_gc2 = Item_GraphConvolution(embedding_size, embedding_size)
        self.item_gc3 = Item_GraphConvolution(embedding_size, embedding_size)

        self.xent_loss = True
        self.T = T
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features, adj, train_set, epoch):

        item_latent = torch.relu(self.nn_emb(features))
        item_latent, loss2 = self.item_gc1(item_latent, adj)
        key_emb = item_latent[train_set[:, 0]]
        pos_emb = item_latent[train_set[:, 1]]
        neg_emb = item_latent[train_set[:, 2:]]

        #pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
        pos_scores = torch.cosine_similarity(key_emb, pos_emb, dim=1)
        #neg_scores = torch.sum(torch.mul(key_emb, neg_emb), dim=1)
        #neg_scores = torch.cosine_similarity(key_emb, neg_emb, dim=1)

        #loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))


        key_emb_repeat = key_emb.unsqueeze(dim=1)
        neg_scores = torch.sum(torch.mul(key_emb_repeat, neg_emb), dim=2)
        #neg_scores = torch.cosine_similarity(key_emb_repeat, neg_emb, dim=2)
        #loss = -torch.mean(torch.log(1 + torch.sum(torch.exp(pos_scores.unsqueeze(1) - neg_scores), dim=1)))
        #score_diff = torch.sum(1 + torch.exp(neg_scores- pos_scores.unsqueeze(1)), dim=1)

        #loss = torch.mean(torch.log(score_diff))

        logits_resort = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        logits_resort /= self.T
        labels = torch.zeros(logits_resort.shape[0], dtype=torch.long).to(features.device)
        loss = self.criterion(logits_resort, labels)


        #loss = -torch.mean(torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-9))
        #loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
        #mrr = self._mrr(torch.unsqueeze(pos_scores, 1), torch.unsqueeze(neg_scores, 1))
        #mrr, hr, ndcg = self.metrics_at_k(torch.unsqueeze(pos_scores, 1), torch.unsqueeze(neg_scores, 1), k=1)
        mrr = 0
        hr = 0
        ndcg = 0
        return loss, mrr, hr, ndcg

    def inference(self, features, adj, test_set):

        item_latent = torch.relu(self.nn_emb(features))
        item_latent,_  = self.item_gc1(item_latent, adj)
        key_emb = item_latent[test_set[:, 0]]
        pos_emb = item_latent[test_set[:, 1]]
        neg_emb = item_latent[test_set[:, 2:]]

        pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
        #key_emb_repeat = key_emb.unsqueeze(dim=1).repeat(1, neg_emb.shape[1], 1)
        key_emb_repeat = key_emb.unsqueeze(dim=1)
        neg_scores = torch.sum(torch.mul(key_emb_repeat, neg_emb), dim=2)

        #mrr = self._mrr(torch.unsqueeze(pos_scores, 1), neg_scores)

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

class READ_GAT(nn.Module):

    def __init__(self, feature_size, embedding_size, num_layer, dropout=0.2):

        super(READ_GAT, self).__init__()
        self.dropout = dropout
        self.num_layer = num_layer
        self.nn_emb = nn.Linear(feature_size, embedding_size, bias=True)

        self.item_gc1 = Item_ChebConvolution(embedding_size, embedding_size)
        self.item_gc2 = Item_GraphConvolution(embedding_size, embedding_size)
        self.item_gc3 = Item_GraphConvolution(embedding_size, embedding_size)

        self.xent_loss = True


    def forward(self, features, adj, train_set, epoch):

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
        item_latent  = self.item_gc1(item_latent, adj)
        key_emb = item_latent[test_set[:, 0]]
        pos_emb = item_latent[test_set[:, 1]]
        neg_emb = item_latent[test_set[:, 2:]]

        pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
        #key_emb_repeat = key_emb.unsqueeze(dim=1).repeat(1, neg_emb.shape[1], 1)
        key_emb_repeat = key_emb.unsqueeze(dim=1)
        neg_scores = torch.sum(torch.mul(key_emb_repeat, neg_emb), dim=2)

        #mrr = self._mrr(torch.unsqueeze(pos_scores, 1), neg_scores)

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

        batch_size = scores.shape[0]  # 矩阵的行数
        score_num = scores.shape[1]  # 每行的元素个数
        shuffle_matrix = torch.zeros(batch_size, score_num)
        random.seed(1)
        for i in range(batch_size):
            # 生成从 0 到 B-1 的数组
            arr = torch.arange(score_num)
            # 随机打乱数组
            random.shuffle(arr)
            # 将打乱后的数组赋值给矩阵的当前行
            shuffle_matrix[i] = arr
        labels = torch.gather(labels, 1, shuffle_matrix)
        scores = torch.gather(scores, 1, shuffle_matrix)

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



