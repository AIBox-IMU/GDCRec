import torch.nn as nn
import torch as th
import dgl
import pdb
from tqdm import tqdm
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import dgl.nn as dglnn
import numpy as np
from sklearn.cluster import KMeans
from dgl.nn.pytorch import GATConv

class HANLayer(nn.Module):
    def __init__(self, args, graph, in_size=32, out_size=32, dropout=0.1,layer_num_heads=1):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(len(graph.etypes)):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.etypes=graph.etypes
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h, etypeh   ):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for etype in self.etypes:
                self._cached_coalesced_graph[
                    etype
                ] = dgl.metapath_reachable_graph(g, etype)

        for i, etype in enumerate(self.etypes):
            new_g = self._cached_coalesced_graph[etype]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = th.stack(
            semantic_embeddings, dim=1
        )

        return self.semantic_embeddings(semantic_embeddings)
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = th.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(1)

