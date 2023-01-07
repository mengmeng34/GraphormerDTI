import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from molecules_graph_transformer.graph_transformer_edge_layer import GraphTransformerLayer


class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params.num_atom_type
        num_bond_type = net_params.num_bond_type
        num_in_degree = net_params.num_in_degree
        num_out_degree = net_params.num_out_degree
        num_spatial_pos = net_params.num_spatial_pos
        hidden_dim = net_params.hidden_dim
        num_heads = net_params.n_heads
        out_dim = net_params.out_dim
        in_feat_dropout = net_params.in_feat_dropout
        dropout = net_params.dropout
        n_layers = net_params.L
        self.layer_norm = net_params.layer_norm
        self.batch_norm = net_params.batch_norm
        self.residual = net_params.residual
        
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)
        
        self.embedding_in_degree = nn.Embedding(num_in_degree, hidden_dim)
        
        self.embedding_out_degree = nn.Embedding(num_out_degree, hidden_dim)

        self.embedding_spatial_pos = nn.Embedding(num_spatial_pos, hidden_dim)

        self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)

        self.e_dis_encoder = nn.Embedding(3*hidden_dim*hidden_dim, 1)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        
    def forward(self, g, h, e, in_degree, out_degree, spatial_pos):

        # input embedding
        h = self.embedding_h(h)
        h = h + self.embedding_in_degree(in_degree) + self.embedding_out_degree(out_degree)
        h = self.in_feat_dropout(h)
        
        e = self.embedding_e(e)
        e_flat = e.permute(1, 0, 2)
        e_flat = torch.bmm(e_flat, self.e_dis_encoder.weight.reshape(-1, 160, 160))
        # e = e_flat.sum(0)
        e = e_flat.sum(0) / (spatial_pos.float().unsqueeze(-1))

        spatial_pos = self.embedding_spatial_pos(spatial_pos)
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e, spatial_pos)
        # g.ndata['h'] = h

        return h
        
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss
