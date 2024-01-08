import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from sklearn.metrics import ndcg_score

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
        self.nn_cat = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, query_x, x):

        batch_size = x.size(0)

        #pair-wise attention
        query = self.query1(query_x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        key = self.key1(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        value = self.value1(x).view(batch_size, -1, self.hidden_size)  # [batch_size, seq_len, hidden_size]

        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        attention_scores = self.softmax(attention_scores / (self.hidden_size ** 0.5))  # [batch_size, seq_len, seq_len]
        x = torch.bmm(attention_scores, value)  # [batch_size, seq_len, hidden_size]

        #self attention
        query = self.query2(x).view(batch_size, -1, self.hidden_size)
        key = self.key2(x).view(batch_size, -1, self.hidden_size)
        value = self.value2(x).view(batch_size, -1, self.hidden_size)

        attention_scores = torch.bmm(query, key.transpose(1, 2))
        attention_scores = self.softmax(attention_scores / (self.hidden_size ** 0.5))
        x = torch.bmm(attention_scores, value)

        x1 = x[:,0,:]
        x2 = x[:,1,:]
        x = torch.cat([x1, x2], dim=1)
        x = self.nn_cat(x)

        return x

class GCN_Low(nn.Module):

    def __init__(self, features_size, embedding_size, bias=False):

        super(GCN_Low, self).__init__()
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

        output = torch.spmm(adj, feature)
        output = 0.5 * output + 0.5 * feature
        output = torch.mm(output, self.weight)

        if self.bias is not None:
            output += self.bias
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

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj):

        output = torch.spmm(adj, feature)
        output = torch.spmm(adj, output)
        output = 0.5 * output - 0.5 * feature
        output = torch.mm(output, self.weight)
        if self.bias is not None:
            output += self.bias

        return output

class Item_Graph_Convolution(nn.Module):

    def __init__(self, features_size, embedding_size, mode):
        super(Item_Graph_Convolution, self).__init__()
        self.mode = mode
        self.gcn_low = GCN_Low(features_size, embedding_size)
        self.gcn_mid = GCN_Mid(features_size, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        if mode == "concat":
            self.nn_cat = nn.Linear(2 * embedding_size, embedding_size)
        else:
            self.nn_cat = None

    def forward(self, feature, adj):

        output_low = self.bn1(self.gcn_low(feature, adj))
        output_mid = self.bn2(self.gcn_mid(feature, adj))

        if self.mode == "att":
            output = torch.cat([torch.unsqueeze(output_low, dim=1), torch.unsqueeze(output_mid, dim=1)], dim=1)
        elif self.mode == "concat":
            output = (self.nn_cat(torch.cat([output_low, output_mid], dim=1)))
        elif self.mode == "mid":
            output = output_mid
        else:
            output = output_low

        return output

class SComGNN(nn.Module):

    def __init__(self, embedding_size, price_n_bins, mode, category_emb_size=768):
        super(SComGNN, self).__init__()
        self.category_emb_size = category_emb_size
        self.mode = mode

        self.embedding_cid2 = nn.Linear(category_emb_size, embedding_size, bias=True)
        self.embedding_cid3 = nn.Linear(category_emb_size, embedding_size, bias=True)
        self.embedding_price = nn.Embedding(price_n_bins, embedding_size)
        self.nn_emb = nn.Linear(embedding_size * 3, embedding_size)
        self.item_gc = Item_Graph_Convolution(embedding_size, embedding_size, self.mode)
        self.two_att = Twostage_Attention(embedding_size)

    def forward(self, features, price, adj, train_set):

        # obtain item embeddings
        cid2 = features[:,:self.category_emb_size]
        cid3 = features[:,self.category_emb_size:]
        embedded_cid2 = self.embedding_cid2(cid2)
        embedded_cid3 = self.embedding_cid3(cid3)
        embed_price = self.embedding_price(price)
        item_latent = torch.relu(self.nn_emb(torch.cat([embedded_cid2, embedded_cid3, embed_price], dim=1)))

        item_latent = self.item_gc(item_latent, adj)

        key_emb = item_latent[train_set[:, 0]]
        pos_emb = item_latent[train_set[:, 1]]
        neg_emb = item_latent[train_set[:, 2:]]

        if self.mode == "att":
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

        else:

            pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
            key_emb = key_emb.unsqueeze(dim=1)
            neg_scores = torch.sum(torch.mul(key_emb, neg_emb), dim=2)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-9))

        return loss

    def inference(self, features, price, adj, test_set):

        cid2 = features[:,:self.category_emb_size]
        cid3 = features[:,self.category_emb_size:]
        # concatenate the three types of embedding
        embedded_cid2 = self.embedding_cid2(cid2)
        embedded_cid3 = self.embedding_cid3(cid3)
        embed_price = self.embedding_price(price)
        item_latent = torch.relu(self.nn_emb(torch.cat([embedded_cid2, embedded_cid3, embed_price], dim=1)))
        item_latent = self.item_gc(item_latent, adj)

        key_emb = item_latent[test_set[:, 0]]
        pos_emb = item_latent[test_set[:, 1]]
        neg_emb = item_latent[test_set[:, 2:]]

        if self.mode == "att":
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

        else:

            pos_scores = torch.sum(torch.mul(key_emb, pos_emb), dim=1)
            key_emb = key_emb.unsqueeze(dim=1)
            neg_scores = torch.sum(torch.mul(key_emb, neg_emb), dim=2)

        hr5, hr10, ndcg = self.metrics(torch.unsqueeze(pos_scores, 1), neg_scores)

        return hr5, hr10, ndcg

    def metrics(self, pos_scores, neg_scores):

        # concatenate the scores of both positive and negative samples
        scores = torch.cat([pos_scores, neg_scores], dim=1)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=1).to(scores.device)
        scores = torch.squeeze(scores)
        labels = torch.squeeze(labels)
        ranking = torch.argsort(scores, dim=1, descending=True)

        #obtain ndcg scores
        ndcg = ndcg_score(labels.cpu(), scores.cpu())

        #obtain hr scores
        k_list = [5, 10]
        hr_list = []
        for k in k_list:
            ranking_k = ranking[:, :k]
            hr = torch.mean(torch.sum(torch.gather(labels, 1, ranking_k), dim=1))
            hr_list.append(hr)

        return hr_list[0], hr_list[1], ndcg