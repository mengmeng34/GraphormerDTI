import math
import random
import os
import pickle
from model import AttentionDTI
from proteins import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from hyperparameter import hyperparameter
from drugparameter import drugparameter
from pytorchtools import EarlyStopping
import timeit
import time
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc

from molecules import MoleculeDataset, MoleculeDatasetDGL, MoleculeDGL
from molecules_graph_transformer.graph_transformer_net import GraphTransformerNet


def show_result(DATASET, lable, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    print("The {} model's results:".format(lable))
    with open("./{}_drug_inductive_setting/results.txt".format(DATASET), 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')

    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))


def load_tensor(file_name, dtype):
    # return [dtype(d).to(hp.device) for d in np.load(file_name + '.npy', allow_pickle=True)]
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_precess(model, pbar, drug_load, LOSS):
    model.eval()
    model_drug.eval()
    test_losses = []
    Y, P, S = [], [], []
    dataloader_iterator = iter(drug_load)
    with torch.no_grad():
        for i, data in pbar:
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                break
            batch_graphs, num_atoms, in_degrees, out_degrees, spatial_pos, edge_inputs = batch
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat']  # num x feat
            edge_inputs = edge_inputs.to(device)
            num_atoms = num_atoms.to(device)
            in_degrees = in_degrees.to(device)
            out_degrees = out_degrees.to(device)
            spatial_pos = spatial_pos.to(device)

            compounds = model_drug.forward(batch_graphs, batch_x, edge_inputs, in_degrees, out_degrees, spatial_pos)

            y = [x for x in compounds.split(list(num_atoms), dim=0)]
            drug_embed = nn.Embedding(65, 160, padding_idx=0, device='cuda')
            outt = []
            for m in range(len(num_atoms)):
                drugem = torch.zeros(150 - num_atoms[m].long(), dtype=torch.long, device='cuda')
                drugembed = drug_embed(drugem)
                out = torch.cat([y[m], drugembed], dim=0)
                outt.append(out)
            compounds = torch.stack(outt, dim=0)

            proteins, labels = data
            proteins = proteins.to(device)
            labels = labels.to(device)

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            predicted_scores = model(compounds, num_atoms, proteins)
            
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Reacll = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)
    return Y, P, test_loss, Accuracy, Precision, Reacll, AUC, PRC


def test_model(dataset_load, drug_load, save_path, DATASET, LOSS, dataset="Train", lable="best", save=True):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
        test_precess(model, test_pbar, drug_load, LOSS)
    if save:
        with open(save_path + "/{}_{}_{}_prediction.txt".format(DATASET, dataset, lable), 'w') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(lable, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)
    
    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


import os
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == "__main__":
    """select seed"""
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(torch.cuda.get_device_capability(device=None), torch.cuda.get_device_name(device=None))

    """init hyperparameters"""
    hp = hyperparameter()
    net_params = drugparameter()

    """Load proteins data."""
    DATASET = "DrugBank"
    # DATASET = "Davis"
    # DATASET = "KIBA"
    print("Train in " + DATASET)
    if DATASET == "DrugBank":
        weight_CE = None
        dir_input = ('./protein_data/{}.txt'.format(DATASET))
        with open(dir_input, "r") as f:
            dataset = f.read().strip().split('\n')
        print("proteins load finished")
    elif DATASET == "Davis":
        weight_CE = torch.FloatTensor([0.3, 0.7]).to(device)
        dir_input = ('./protein_data/{}.txt'.format(DATASET))
        with open(dir_input, "r") as f:
            dataset = f.read().strip().split('\n')
        print("proteins load finished")
    elif DATASET == "KIBA":
        weight_CE = torch.FloatTensor([0.2, 0.8]).to(device)
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

    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []
    
    TVdata, train_dataset, valid_dataset, test_dataset = [], [], [], []
    for i in dataset:
        pair = i.strip().split()
        drug_id, protein_id, proteinstr, label = pair[-4], pair[-3], pair[-2], pair[-1]
        if drug_id in train_drug:
            TVdata.append(i)
        if drug_id in test_drug:
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
        
    g = torch.Generator()
    g.manual_seed(0)
    train_dataset_load = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, generator=g)
    valid_dataset_load = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_dataset_load = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    drug_dataset = MoleculeDataset(DATASET + '_drug_inductive_setting_' + str(SEED))
    print("molecules load finished")
        
    g = torch.Generator()
    g.manual_seed(0)
    train_drug_dataset, valid_drug_dataset, test_drug_dataset = drug_dataset.train, drug_dataset.val, drug_dataset.test
    train_drug_load = DataLoader(train_drug_dataset, batch_size=net_params.Batch_size, shuffle=True, num_workers=0,
                                 collate_fn=drug_dataset.collate, generator=g)
    valid_drug_load = DataLoader(valid_drug_dataset, batch_size=net_params.Batch_size, shuffle=False, num_workers=0,
                                 collate_fn=drug_dataset.collate)
    test_drug_load = DataLoader(test_drug_dataset, batch_size=net_params.Batch_size, shuffle=False, num_workers=0,
                                collate_fn=drug_dataset.collate)

    """ create model"""
    model = AttentionDTI(hp).to(device)
    # model = nn.DataParallel(AttentionDTI(hp), device_ids=[0, 1]).to(device)
    model_drug = GraphTransformerNet(net_params).to(device)
    """weight initialize"""
    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    """load trained model"""
    # self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(
        [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}],
        lr=hp.Learning_rate)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate * 10,
                                            cycle_momentum=False,
                                            step_size_up=len(TVdata)-valid_size // hp.Batch_size)
    Loss = nn.CrossEntropyLoss(weight=weight_CE)
        
    save_path = "./" + DATASET + "_drug_inductive_setting"
    note = ''
    writer = SummaryWriter(log_dir=save_path, comment=note)

    """Output files."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_results = save_path + '/The_results_of_whole_dataset.txt'

    with open(file_results, 'w') as f:
        hp_attr = '\n'.join(['%s:%s' % item for item in hp.__dict__.items()])
        f.write(hp_attr + '\n')

    early_stopping = EarlyStopping(savepath=save_path, patience=hp.Patience, verbose=True, delta=0)
    # print("Before train,test the model:")
    # _,_,_,_,_,_ = test_model(test_dataset_load, save_path, DATASET, Loss, dataset="Test",lable="untrain",save=False)
    
    """Start training."""
    print('Training...')
    start = timeit.default_timer()
        
    for epoch in range(1, hp.Epoch + 1):
        torch.cuda.empty_cache()
        trian_pbar = tqdm(
            enumerate(
                BackgroundGenerator(train_dataset_load)),
            total=len(train_dataset_load))

        """train"""
        train_losses_in_epoch = []
        # train_losses_in_epoch = 0
        model.train()
        model_drug.train()
            
        dataloader_iterator = iter(train_drug_load)
        for trian_i, train_data in trian_pbar:
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                break
            batch_graphs, num_atoms, in_degrees, out_degrees, spatial_pos, edge_inputs = batch
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat']  # num x feat
            edge_inputs = edge_inputs.to(device)
            num_atoms = num_atoms.to(device)
            in_degrees = in_degrees.to(device)
            out_degrees = out_degrees.to(device)
            spatial_pos = spatial_pos.to(device)
            optimizer.zero_grad()
                
            trian_compounds = model_drug.forward(batch_graphs, batch_x, edge_inputs, in_degrees, out_degrees, spatial_pos)

            y = [x for x in trian_compounds.split(list(num_atoms), dim=0)]
            drug_embed = nn.Embedding(65, 160, padding_idx=0, device='cuda')
            outt = []
            for m in range(len(num_atoms)):
                drugem = torch.zeros(150 - num_atoms[m].long(), dtype=torch.long, device='cuda')
                drugembed = drug_embed(drugem)
                out = torch.cat([y[m], drugembed], dim=0)
                outt.append(out)
            trian_compounds = torch.stack(outt, dim=0)
                
            trian_proteins, trian_labels = train_data
            trian_proteins = trian_proteins.to(device)
            trian_labels = trian_labels.to(device)

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            optimizer.zero_grad()
            predicted_interaction = model(trian_compounds, num_atoms, trian_proteins)
            train_loss = Loss(predicted_interaction, trian_labels)    # 1
            train_losses_in_epoch.append(train_loss.item())
            # train_losses_in_epoch += train_loss.item()
            train_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
            optimizer.step()
            scheduler.step()
            
        train_loss_a_epoch = np.average(train_losses_in_epoch)
        # train_loss_a_epoch = train_losses_in_epoch/math.ceil(len(train_drug_dataset)/hp.Batch_size)
        writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)
        # avg_train_losses.append(train_loss_a_epoch)

        # valid
        valid_pbar = tqdm(
            enumerate(
                BackgroundGenerator(valid_dataset_load)),
            total=len(valid_dataset_load))
        # loss_dev, AUC_dev, PRC_dev = valider.valid(valid_pbar,weight_CE)
        valid_losses_in_epoch = []
        model.eval()
        model_drug.eval()
        Y, P, S = [], [], []
        dataloader_iterator = iter(valid_drug_load)
        with torch.no_grad():
            for valid_i, valid_data in valid_pbar:
                try:
                    batch = next(dataloader_iterator)
                except StopIteration:
                    break
                batch_graphs, num_atoms, in_degrees, out_degrees, spatial_pos, edge_inputs = batch
                batch_graphs = batch_graphs.to(device)
                batch_x = batch_graphs.ndata['feat']  # num x feat
                edge_inputs = edge_inputs.to(device)
                num_atoms = num_atoms.to(device)
                in_degrees = in_degrees.to(device)
                out_degrees = out_degrees.to(device)
                spatial_pos = spatial_pos.to(device)
                
                valid_compounds = model_drug.forward(batch_graphs, batch_x, edge_inputs, in_degrees, out_degrees, spatial_pos)

                y = [x for x in valid_compounds.split(list(num_atoms), dim=0)]
                drug_embed = nn.Embedding(65, 160, padding_idx=0, device='cuda')
                outt = []
                for m in range(len(num_atoms)):
                    drugem = torch.zeros(150 - num_atoms[m].long(), dtype=torch.long, device='cuda')
                    drugembed = drug_embed(drugem)
                    out = torch.cat([y[m], drugembed], dim=0)
                    outt.append(out)
                valid_compounds = torch.stack(outt, dim=0)

                valid_proteins, valid_labels = valid_data
                valid_proteins = valid_proteins.to(device)
                valid_labels = valid_labels.to(device)

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                valid_scores = model(valid_compounds, num_atoms, valid_proteins)
                valid_loss = Loss(valid_scores, valid_labels)
                valid_labels = valid_labels.to('cpu').data.numpy()
                valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                valid_predictions = np.argmax(valid_scores, axis=1)
                valid_scores = valid_scores[:, 1]

                valid_losses_in_epoch.append(valid_loss.item())
                Y.extend(valid_labels)
                P.extend(valid_predictions)
                S.extend(valid_scores)
            
        Precision_dev = precision_score(Y, P)
        Reacll_dev = recall_score(Y, P)
        Accuracy_dev = accuracy_score(Y, P)
        AUC_dev = roc_auc_score(Y, S)
        tpr, fpr, _ = precision_recall_curve(Y, S)
        PRC_dev = auc(fpr, tpr)
        valid_loss_a_epoch = np.average(valid_losses_in_epoch)
        # avg_valid_loss.append(valid_loss)

        epoch_len = len(str(hp.Epoch))

        print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                     f'train_loss: {train_loss_a_epoch:.5f} ' +
                     f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                     f'valid_AUC: {AUC_dev:.5f} ' +
                     f'valid_PRC: {PRC_dev:.5f} ' +
                     f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                     f'valid_Precision: {Precision_dev:.5f} ' +
                     f'valid_Reacll: {Reacll_dev:.5f} ')

        writer.add_scalar('Valid Loss', valid_loss_a_epoch, epoch)
        writer.add_scalar('Valid AUC', AUC_dev, epoch)
        writer.add_scalar('Valid AUPR', PRC_dev, epoch)
        writer.add_scalar('Valid Accuracy', Accuracy_dev, epoch)
        writer.add_scalar('Valid Precision', Precision_dev, epoch)
        writer.add_scalar('Valid Reacll', Reacll_dev, epoch)
        writer.add_scalar('Learn Rate', optimizer.param_groups[0]['lr'], epoch)

        print(print_msg)
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss_a_epoch, model_drug, model, epoch)

    trainset_test_stable_results, _, _, _, _, _ = test_model(train_dataset_load, train_drug_load, save_path, DATASET, Loss, dataset="Train", lable="stable")
    validset_test_stable_results, _, _, _, _, _ = test_model(valid_dataset_load, valid_drug_load, save_path, DATASET, Loss, dataset="Valid", lable="stable")
    testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
        test_model(test_dataset_load, test_drug_load, save_path, DATASET, Loss, dataset="Test", lable="stable")
    AUC_List_stable.append(AUC_test)
    Accuracy_List_stable.append(Accuracy_test)
    AUPR_List_stable.append(PRC_test)
    Recall_List_stable.append(Recall_test)
    Precision_List_stable.append(Precision_test)
    with open(save_path + "/The_results_of_whole_dataset.txt", 'a') as f:
        f.write("Test the stable model" + '\n')
        f.write(trainset_test_stable_results + '\n')
        f.write(validset_test_stable_results + '\n')
        f.write(testset_test_stable_results + '\n')

    show_result(DATASET, "stable",
                Accuracy_List_stable, Precision_List_stable, Recall_List_stable,
                AUC_List_stable, AUPR_List_stable)
