import torch
import pickle
import torch.utils.data
import time
import os
import csv
import dgl
from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib
from tkinter import _flatten


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs

        with open(data_dir + "/%s.pickle" % self.split, "rb") as f:
            self.data = pickle.load(f)

        self.graph_lists = []
        self.num_atoms = []
        self.in_degree = []
        self.out_degree = []
        self.edge_input = []
        self.n_samples = len(self.data)
        self._prepare()

    def _prepare(self):

        for molecule in self.data:
            node_features = molecule['atom_type'].long()

            adj = torch.sum(molecule['edge_input'], dim=2)
            edge_list = (adj != 0).nonzero()

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features

            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            edge = torch.stack([molecule['edge_input'][src, dst, :] for src, dst in edge_list]).long()
            # g.edata['feat'] = edge_features

            self.graph_lists.append(g)
            self.num_atoms.append(molecule['num_atom'])
            self.in_degree.append(molecule['in_degree'])
            self.out_degree.append(molecule['out_degree'])
            self.edge_input.append(edge)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.num_atoms[idx], self.in_degree[idx], self.out_degree[idx], self.spatial_pos[idx], self.edge_input[idx]

    
class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Zinc'):
        t0 = time.time()
        self.name = name

        self.num_atom_type = 10
        self.num_bond_type = 3

        data_dir = './data/molecules'

        if self.name == 'ZINC-full':
            data_dir = './data/molecules/zinc_full'
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=220011)
            self.val = MoleculeDGL(data_dir, 'val', num_graphs=24445)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=5000)
        else:
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=10000)
            self.val = MoleculeDGL(data_dir, 'val', num_graphs=1000)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=1000)
        print("Time taken: {:.4f}s".format(time.time() - t0))
    

class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        data_dir = './molecule_data/'
        with open(data_dir + name + '.pkl', "rb") as f:
            f1 = pickle.load(f)
            f.close()
            self.train = f1.train
            self.val = f1.val
            self.test = f1.test
        print('train, val, test sizes :', len(self.train), len(self.val), len(self.test))

    def collate(self, samples):
        # graphs, nums = map(list, zip(*samples))
        graphs, nums, in_degree, out_degree, spatial_pos, edge_input = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        nums = torch.tensor(np.array(nums)).unsqueeze(1)
        in_degree = torch.tensor(np.array(list(_flatten(in_degree))))
        out_degree = torch.tensor(np.array(list(_flatten(out_degree))))
        spatial_pos = torch.cat(list(spatial_pos))
        edge_input = torch.cat(edge_input, dim=0)
        
        return batched_graph, nums, in_degree, out_degree, spatial_pos, edge_input

