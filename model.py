import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDTI(nn.Module):
    def __init__(self, hp,
                 protein_MAX_LENGH=1000, drug_MAX_LENGH=150):
        super(AttentionDTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.protein_kernel = hp.protein_kernel

        self.drug_embed = nn.Embedding(65, 160, padding_idx=0)
        self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        self.Protein_max_pool = nn.MaxPool1d(
            self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)
        self.Drug_max_pool = nn.MaxPool1d(self.drug_MAX_LENGH)
        self.attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.protein_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.drug_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv * 8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, num_atoms, protein):

        drug_masks = []
        for m in range(len(num_atoms)):
            drug_mask = torch.cat([torch.ones(num_atoms[m].long(), dtype=torch.long, device=num_atoms.device),
                                   torch.zeros(150 - num_atoms[m].long(), dtype=torch.long, device=num_atoms.device)], dim=0)
            drug_masks.append(drug_mask)

        drugConv = drug.permute(0, 2, 1)

        proteinembed = self.protein_embed(protein)
        proteinembed = proteinembed.permute(0, 2, 1)
        proteinConv = self.Protein_CNNs(proteinembed)
        
        drug_att = self.drug_attention_layer(drugConv.permute(0, 2, 1))
        drug_masks = torch.stack(drug_masks, dim=0)
        drug_padding_mask = torch.where(drug_masks > 0, True, False)
        drug_padding_mask = torch.unsqueeze(drug_padding_mask, 2).repeat(1, 1, drug_att.shape[-1])
        drug_att = drug_att * drug_padding_mask
        del drug_masks, drug_padding_mask
        # drugConv = drugConv * drug_padding.permute(0, 2, 1)
        
        protein_att = self.protein_attention_layer(proteinConv.permute(0, 2, 1))
        protein_padding_mask = torch.where(protein > 0, True, False)[:, 0:979]
        protein_padding_mask = torch.unsqueeze(protein_padding_mask, 2).repeat(1, 1, protein_att.shape[-1])
        protein_att = protein_att * protein_padding_mask
        del protein, protein_padding_mask
        # proteinConv = proteinConv * protein_padding.permute(0, 2, 1)

        drug_att = torch.unsqueeze(drug_att, 2).repeat(1, 1, proteinConv.shape[-1], 1)  # repeat along protein size
        protein_att = torch.unsqueeze(protein_att, 1).repeat(1, drugConv.shape[-1], 1, 1)  # repeat along drug size
        Atten_matrix = self.attention_layer(self.relu(drug_att + protein_att))
        Compound_atte = torch.mean(Atten_matrix, 2)
        Protein_atte = torch.mean(Atten_matrix, 1)
        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1))
        Protein_atte = self.sigmoid(Protein_atte.permute(0, 2, 1))

        drugConv = drugConv * 0.5 + drugConv * Compound_atte
        proteinConv = proteinConv * 0.5 + proteinConv * Protein_atte

        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)
        
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)

        return predict
        
        