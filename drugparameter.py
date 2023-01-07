import torch
from datetime import datetime


class drugparameter():
    def __init__(self):
        self.L = 12
        self.n_heads = 8
        self.hidden_dim = 160
        self.out_dim = 160
        self.residual = True
        self.in_feat_dropout = 0.0
        self.dropout = 0.0
        self.layer_norm = False
        self.batch_norm = True
        self.pos_enc_dim = 8
        self.num_atom_type = 118
        self.num_bond_type = 4
        self.num_in_degree = 10
        self.num_out_degree = 10
        self.num_spatial_pos = 4
        self.Batch_size = 32
