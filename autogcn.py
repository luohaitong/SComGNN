import torch
import torch.nn as nn
import numpy as np
from dgl.nn.pytorch import APPNPConv
import dgl.function as fn
from torch.nn import init


class MPConv(nn.Module):
    def __init__(self):
        super(MPConv, self).__init__()

    def forward(self, graph, feat):
        graph = graph.local_var()
        norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = torch.reshape(norm, shp).to(feat.device)

        # normalization by src node
        feat = feat * norm
        graph.ndata['h'] = feat
        graph.update_all(fn.copy_src(src='h', out='m'),
                         fn.sum(msg='m', out='h'))

        feat = graph.ndata['h']
        # normalization by dst node
        feat = feat * norm
        return feat


class AUTOGCNLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """

    def __init__(self, in_dim, out_dim, activation, dropout, graph_norm, batch_norm, K=8, num_filters=1, residual=False, gate=True, opt='over'):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.gate = gate
        if in_dim != out_dim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.num_filters = num_filters
        self.low = nn.ModuleList()
        self.high = nn.ModuleList()
        self.mid = nn.ModuleList()
        self.low_gamma = nn.ParameterList()
        self.mid_gamma = nn.ParameterList()
        self.high_gamma = nn.ParameterList()
        self.low_alpha = nn.ParameterList()
        self.mid_alpha = nn.ParameterList()
        self.high_alpha = nn.ParameterList()

        for i in range(self.num_filters):
            self.low.append(nn.Linear(in_dim, out_dim,bias=False))
            self.high.append(nn.Linear(in_dim, out_dim,bias=False))
            self.mid.append(nn.Linear(in_dim, out_dim,bias=False))

        self.eps = 1e-9
        self.K = K
        self.mp = MPConv()
        self.opt = opt

        if self.opt == 'over':
            for i in range(self.num_filters):
                self.low_gamma.append(torch.nn.Parameter(torch.FloatTensor([1/K for i in range(K)])))
                self.mid_gamma.append(torch.nn.Parameter(torch.FloatTensor([1/K for i in range(K)])))
                self.high_gamma.append(torch.nn.Parameter(torch.FloatTensor([1/K for i in range(K)])))
            self.alpha = torch.Tensor(np.linspace(-self.eps, 1+self.eps, self.K))
            self.midalpha = torch.Tensor(np.linspace(-self.eps, 1+self.eps, self.K))

        elif self.opt == 'single':
            self.lowalpha = torch.nn.Parameter(torch.FloatTensor([0 for i in range(self.num_filters)]))
            self.lowgamma = torch.nn.Parameter(torch.FloatTensor([1 for i in range(self.num_filters)]))
            self.highalpha = torch.nn.Parameter(torch.FloatTensor([0 for i in range(self.num_filters)]))
            self.highgamma = torch.nn.Parameter(torch.FloatTensor([1 for i in range(self.num_filters)]))
            self.midalpha = torch.nn.Parameter(torch.FloatTensor([0 for i in range(self.num_filters)]))
            self.midgamma = torch.nn.Parameter(torch.FloatTensor([1 for i in range(self.num_filters)]))

        self.bias = nn.Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            init.zeros_(self.bias)


    def forward(self, g, feature, snorm_n):
        h = self.mp(g, feature)
        out_low = []
        out_mid = []
        out_high = []

        h1 = self.mp(g, h)
        if self.opt== 'over':
            alpha = self.alpha.to(feature.device)
            midalpha = self.midalpha.to(feature.device)
            for i in range(self.num_filters):
                gamma = self.low_gamma[i]
                gamma = torch.relu(gamma)
                gamma = gamma.squeeze()
                a = torch.dot(alpha, gamma)
                b = torch.dot(1 - alpha, gamma)
                o = a * h + b * feature
                o = self.low[i](o)
                out_low.append(o)

                gamma = self.high_gamma[i]
                gamma = torch.relu(gamma)
                gamma = gamma.squeeze()
                a = torch.dot(-alpha, gamma)
                b = torch.dot(1 - alpha, gamma)
                o = a * h + b * feature
                o = self.high[i](o)
                out_high.append(o)

                gamma = self.mid_gamma[i]
                gamma = torch.relu(gamma)
                gamma = gamma.squeeze()
                a = torch.sum(gamma)
                c = torch.dot(midalpha, gamma)
                o = a * h1 - c * feature
                o = self.mid[i](o)
                out_mid.append(o)

        elif self.opt == 'single':
                lowalpha = torch.nn.functional.hardtanh(self.lowalpha, min_val=0., max_val=1.)
                lowgamma = torch.relu(self.lowgamma)
                highalpha = torch.nn.functional.hardtanh(self.highalpha, min_val=0., max_val=1.)
                highgamma = torch.relu(self.highgamma)
                midalpha = torch.nn.functional.hardtanh(self.midalpha, min_val=0., max_val=1.)
                midgamma = torch.relu(self.midgamma)
                h1 = self.mp(g, h)
                for i in range(self.num_filters):
                    o = (lowalpha[i] * h + (1 - lowalpha[i]) * feature) * lowgamma[i]
                    o = self.low[i](o)
                    out_low.append(o)
                    o = (-highalpha[i] * h + (1 - highalpha[i]) * feature) * highgamma[i]
                    o = self.high[i](o)
                    out_mid.append(o)
                    o = (h1 - midalpha[i] * feature) * midgamma[i]
                    o = self.mid[i](o)
                    out_high.append(o)

        elif self.opt == 'fix':
            h1 = self.mp(g, h)
            for i in range(self.num_filters):
                o = 0.5 * h + 0.5 * feature
                o = self.low[i](o)
                out_low.append(o)
                o = -0.5 * h + 0.5 * feature
                o = self.high[i](o)
                out_mid.append(o)
                o = (h1 - 0.5 * feature)
                o = self.mid[i](o)
                out_high.append(o)


        # if self.gate:
        for i in range(self.num_filters):
            out_low[i] = out_low[i] * (torch.sigmoid(out_high[i] + out_mid[i]))
            out_mid[i] = out_mid[i] * (torch.sigmoid(out_low[i] + out_high[i]))
            out_high[i] = out_high[i] * (torch.sigmoid(out_mid[i] + out_low[i]))
            # out_low[i] = out_low[i] * (torch.sigmoid(out_mid[i] + out_high[i]))
            # out_mid[i] = out_mid[i] * (torch.sigmoid(out_high[i] + out_low[i]))
            # out_high[i] = out_high[i] * (torch.sigmoid(out_low[i] + out_mid[i]))
        out = []
        out.extend(out_low)
        out.extend(out_mid)
        out.extend(out_high)
        out = [o.unsqueeze(0) for o in out]
        out = torch.cat(out, dim=0)
        out = torch.sum(out, dim=0)
        out = out.squeeze()
        out = out + self.bias


        if self.graph_norm:
            out = out * snorm_n  # normalize activation w.r.t. graph size

        if self.batch_norm:
            out = self.batchnorm_h(out)  # batch normalization

        if self.activation:
            out = self.activation(out)

        if self.residual:
            out = feature + out  # residual connection
        out = self.dropout(out)
        return out

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                                                         self.in_channels,
                                                                         self.out_channels, self.residual)