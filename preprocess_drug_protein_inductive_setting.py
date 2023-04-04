import math
import random
import os
import pickle
import linecache
import csv
import dgl
import torch.utils.data
from scipy import sparse as sp
import networkx as nx
import hashlib
from proteins import CustomDataSet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from molecules import MoleculeDataset, MoleculeDGL


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def save_dict(obj, name):
    with open(data_dir + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(DATASET, name):
    with open('./' + DATASET + '_drugs/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
data_dir = './molecule_data/'


def generate_data(DATASET, data):
    data_len = len(data)
    
    drug_ids = []
    for i in range(data_len):
        pair = data[i]
        pair = pair.strip().split()
        drug_id, protein_id, proteinstr, label = pair[-4], pair[-3], pair[-2], pair[-1]
        drug_ids.append(drug_id)
    drugs = []
    for v in range(data_len):
        filename = drug_ids[v]
        d = load_dict(DATASET, filename)
        drugs.append(d)
    print(len(drugs), 'finished')
    return drugs


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.graph_lists = []
        self.num_atoms = []
        self.in_degree = []
        self.out_degree = []
        self.spatial_pos = []
        self.edge_input = []
        self.n_samples = len(self.data)
        self._prepare()

    def _prepare(self):
        for molecule in self.data:
            node_features = molecule['atom_type'].long()

            spatial_pos_list = (molecule['spatial_pos'] != 0).nonzero()
            spatial = torch.stack([molecule['spatial_pos'][src, dst] for src, dst in spatial_pos_list])

            adj = torch.sum(molecule['edge_input'], dim=2)
            edge_list = (adj != 0).nonzero()

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features

            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            edge = torch.stack([molecule['edge_input'][src, dst, :] for src, dst in edge_list])
            
            self.graph_lists.append(g)
            self.num_atoms.append(molecule['num_atom'])
            self.in_degree.append(molecule['in_degree'])
            self.out_degree.append(molecule['out_degree'])
            self.spatial_pos.append(spatial)
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
    def __init__(self, train, valid, test):

        self.train = MoleculeDGL(train)
        print('train dataset finished')
        self.val = MoleculeDGL(valid)
        print('valid dataset finished')
        self.test = MoleculeDGL(test)
        print('test dataset finished')


if __name__ == "__main__":
    """select seed"""
    SEED = 0
    random.seed(SEED)
    torch.manual_seed(SEED)

    """Load proteins data."""
    DATASET = "DrugBank"
    # DATASET = "Davis"
    # DATASET = "KIBA"
    print("Train in " + DATASET)
    if DATASET == "DrugBank":
        dir_input = ('./protein_data/{}.txt'.format(DATASET))
        with open(dir_input, "r") as f:
            dataset = f.read().strip().split('\n')
        print("proteins load finished")
    elif DATASET == "Davis":
        dir_input = ('./protein_data/{}.txt'.format(DATASET))
        with open(dir_input, "r") as f:
            dataset = f.read().strip().split('\n')
        print("proteins load finished")
    elif DATASET == "KIBA":
        dir_input = ('./protein_data/{}.txt'.format(DATASET))
        with open(dir_input, "r") as f:
            dataset = f.read().strip().split('\n')
        print("proteins load finished")

    # random shuffle
    print("data shuffle")
    dataset = shuffle_dataset(dataset, SEED)
    
    drug_ids, protein_ids = [], []
    for pair in dataset:
        pair = pair.strip().split()
        drug_id, protein_id, proteinstr, label = pair[-4], pair[-3], pair[-2], pair[-1]
        drug_ids.append(drug_id)
        protein_ids.append(protein_id)
    drug_set = list(set(drug_ids))
    drug_set.sort(key=drug_ids.index)
    protein_set = list(set(protein_ids))
    protein_set.sort(key=protein_ids.index)

    torch.manual_seed(SEED)
    train_d, test_d = torch.utils.data.random_split(drug_set, [math.ceil(0.5*len(drug_set)), len(drug_set)-math.ceil(0.5*len(drug_set))])
    train_drug, test_drug = [], []
    for i in range(len(train_d)):
        pair = drug_set[train_d.indices[i]]
        train_drug.append(pair)
    for i in range(len(test_d)):
        pair = drug_set[test_d.indices[i]]
        test_drug.append(pair)

    torch.manual_seed(SEED)
    train_p, test_p = torch.utils.data.random_split(protein_set, [math.ceil(0.5*len(protein_set)), len(protein_set)-math.ceil(0.5*len(protein_set))])
    train_protein, test_protein = [], []
    for i in range(len(train_p)):
        pair = protein_set[train_p.indices[i]]
        train_protein.append(pair)
    for i in range(len(test_p)):
        pair = protein_set[test_p.indices[i]]
        test_protein.append(pair)

    TVdata, train_dataset, valid_dataset, test_dataset = [], [], [], []
    for i in dataset:
        pair = i.strip().split()
        drug_id, protein_id, proteinstr, label = pair[-4], pair[-3], pair[-2], pair[-1]
        if drug_id in train_drug and protein_id in train_protein:
            TVdata.append(i)
        if drug_id in test_drug and protein_id in test_protein:
            test_dataset.append(i)

    valid_size = math.ceil(0.2 * len(TVdata))
    torch.manual_seed(SEED)
    train_data, valid_data = torch.utils.data.random_split(TVdata, [len(TVdata)-valid_size, valid_size])
    
    traindata, validdata = [], []
    for i in range(len(train_data)):
        pair = TVdata[train_data.indices[i]]
        traindata.append(pair)
    for i in range(len(valid_data)):
        pair = TVdata[valid_data.indices[i]]
        validdata.append(pair)
    
    train_dataset = CustomDataSet(traindata)
    valid_dataset = CustomDataSet(validdata)
    test_dataset = CustomDataSet(test_dataset)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    
    """generate train, valid, test datasets"""
    train = generate_data(DATASET, train_dataset.pairs)
    valid = generate_data(DATASET, valid_dataset.pairs)
    test = generate_data(DATASET, test_dataset.pairs)
    
    datasets = MoleculeDatasetDGL(train, valid, test)
    print(len(datasets.train), len(datasets.val), len(datasets.test))
    if not os.path.exists("./molecule_data/"):
        os.mkdir("./molecule_data/")
    save_dict(datasets, DATASET + "_drug_protein_inductive_setting_" + str(SEED))
    