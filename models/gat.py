import torch.nn as nn
import torch as th
import pdb
from tqdm import tqdm
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import dgl.nn as dglnn
import numpy as np
from sklearn.cluster import KMeans

class GatLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.k = args.k
        self.sigma = args.sigma
        self.gamma = args.gamma



        self.attn_weights = nn.Parameter(th.Tensor(32, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attn_weights.data, gain=1.414)


    def similarity_matrix(self, X, sigma = 1.0, gamma = 2.0):
        dists = th.cdist(X, X)
        sims = th.exp(-dists / (sigma * dists.mean(dim = -1).mean(dim = -1).reshape(-1, 1, 1)))
        return sims


    def gat_selection_feature(self, nodes):
        device = nodes.mailbox['m'].device
        feature = nodes.mailbox['m']
        sims = self.similarity_matrix(feature, self.sigma, self.gamma)
        batch_num, neighbor_num, feature_size = feature.shape
        nodes_selected = []
        cache = th.zeros((batch_num, 1, neighbor_num), device=device)
        for i in range(1000000):
            attention = self.attention_weights(feature)
            gain = th.sum(th.maximum(sims, cache) - cache, dim=-1)
            gain=gain * attention
            max_value = gain.max().item()

            selected = th.argmax(gain, dim=1)

            cache = th.maximum(sims[th.arange(batch_num, device=device), selected].unsqueeze(1), cache)

            nodes_selected.append(selected)
            if max_value < 0.35:
                break
        return th.stack(nodes_selected).t()

    def attention_weights(self, features):

        attn_input = th.matmul(features, self.attn_weights).squeeze(-1)

        attn_input = attn_input - attn_input.max(dim=1, keepdim=True)[0]

        attention = F.softmax(attn_input, dim=1)

        return attention

    def sub_reduction(self, nodes):

        mail = nodes.mailbox['m']
        batch_size, neighbor_size, feature_size = mail.shape

        if (-1 in nodes.mailbox['c']) or nodes.mailbox['m'].shape[1] <= self.k:
            mail = mail.sum(dim = 1)
        else:
            neighbors = self.gat_selection_feature(nodes)
            mail = mail[th.arange(batch_size, dtype=th.long, device=mail.device).unsqueeze(-1), neighbors]
            mail = mail.sum(dim=1)
        return {'h': mail}



    def category_aggregation(self, edges):
        return {'c': edges.src['category'], 'm': edges.src['h']}

    def forward(self, graph, h, etype):
        with graph.local_scope():
            src, _ , dst = etype
            feat_src = h[src]
            feat_dst = h[dst]

            degs = graph.out_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = th.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(self.category_aggregation, self.sub_reduction, etype = etype)

            rst = graph.nodes[dst].data['h']

            degs = graph.in_degrees(etype = etype).float().clamp(min = 1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm
            return rst

