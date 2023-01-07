import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN,np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)

def collate_fn(batch_data):
    N = len(batch_data)
    protein_max = 1000
    protein_new = torch.zeros((N, protein_max),dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    for i,pair in enumerate(batch_data):
        # _, _, compoundstr, proteinstr, label = pair.strip().split()
        pair = pair.strip().split()
        drug_id, protein_id, proteinstr, label = pair[-4], pair[-3], pair[-2], pair[-1]
        proteinint = torch.from_numpy(label_sequence(proteinstr, CHARPROTSET, protein_max))
        protein_new[i] = proteinint
        label = float(label)
        labels_new[i] = np.int(label)
    
    return (protein_new, labels_new)

