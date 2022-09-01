import pandas as pd
import numpy as np
import torch
import json
from evaluation import greedy_alignment
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import time
from sklearn import preprocessing

def get_source_and_target_matrices(alignment_dict, entity2vec1, entity2vec2, train_ents, valid_ents, test_ents, return_test_embs=True):
    """This function takes the dictionary of aligned entities between two KGs and their corresponding embeddings (as entity to vector dictionaries)
    and returns S, T, S_test, T_test, and R defined as follows:
    
    -- S: Subset of the source embeddings, i.e. the matrix of aligned entity embeddings in the first knowledge graph
    
    -- T: Subset of the matrix of aligned entity embeddings in the second knowledge graph
    
    -- S_test and T_test are the embedding matrices corresponding to the lef-out SameAs links
    
    """
    print()
    
    if not return_test_embs:
        train_ents = alignment_dict.keys()
    emb_dim = entity2vec1.shape[1]
    S = np.empty((len(train_ents), emb_dim))
    T = np.empty((len(train_ents), emb_dim))
    if return_test_embs:
        S_test = np.empty((len(test_ents), emb_dim))
        T_test = np.empty((len(test_ents), emb_dim))
        S_valid = np.empty((len(valid_ents), emb_dim))
        T_valid = np.empty((len(valid_ents), emb_dim))

    for i, key in tqdm(enumerate(train_ents), total=len(train_ents), desc='Building training alignment matrices...'):
        S[i] = entity2vec1[key] if isinstance(entity2vec1, dict) else entity2vec1.loc[key].values
        T[i] = entity2vec2[alignment_dict[key]] if isinstance(entity2vec2, dict) else entity2vec2.loc[alignment_dict[key]].values
        
    if return_test_embs:
        for i, key in tqdm(enumerate(test_ents), total=len(test_ents), desc='Building test alignment matrices'):
            S_test[i] = entity2vec1[key] if isinstance(entity2vec1, dict) else entity2vec1.loc[key].values
            T_test[i] = entity2vec2[alignment_dict[key]] if isinstance(entity2vec2, dict) else entity2vec2.loc[alignment_dict[key]].values
            
        for i, key in tqdm(enumerate(valid_ents), total=len(valid_ents), desc='Building validation alignment matrices'):
            S_valid[i] = entity2vec1[key] if isinstance(entity2vec1, dict) else entity2vec1.loc[key].values
            T_valid[i] = entity2vec2[alignment_dict[key]] if isinstance(entity2vec2, dict) else entity2vec2.loc[alignment_dict[key]].values
    
    if return_test_embs:
        S, T = preprocessing.normalize(S), preprocessing.normalize(T)
        S_test, T_test = preprocessing.normalize(S_test), preprocessing.normalize(T_test)
        S_valid, T_valid = preprocessing.normalize(S_valid), preprocessing.normalize(T_valid)
        return S, T, S_valid, T_valid, S_test, T_test
    else:
        S, T = preprocessing.normalize(S), preprocessing.normalize(T)
        return S, T
        
        
def get_data(dataset, fold):
    data = pd.read_csv(f"{dataset}/Shallom_entity_embeddings.csv")
    emb1 = data[data['Unnamed: 0'].apply(lambda x: 'http://dbpedia.org' in x)].set_index('Unnamed: 0')
    emb2 = data.iloc[np.setdiff1d(np.arange(data.shape[0]),np.array(emb1.index))].set_index('Unnamed: 0')

    with open(f"{dataset}/KG/ent_links") as file:
        kg1_kg2_links = file.read().strip().split('\n')
        kg1_kg2_links = dict([line.split('\t') for line in kg1_kg2_links])

    with open(f"{dataset}/KG/721_5fold/{fold}/test_links") as file:
        test_ents = file.read().strip().split('\n')
    test_ents = [line.split('\t')[0] for line in test_ents]
    
    with open(f"{dataset}/KG/721_5fold/{fold}/train_links") as file:
        train_ents = file.read().strip().split('\n')
    train_ents = [line.split('\t')[0] for line in train_ents]
    
    with open(f"{dataset}/KG/721_5fold/{fold}/valid_links") as file:
        valid_ents = file.read().strip().split('\n')
    valid_ents = [line.split('\t')[0] for line in valid_ents]
    
    return emb1, emb2, train_ents, valid_ents, test_ents, kg1_kg2_links


def generate_false_matching(data):
    corrupted_data = []
    source, target = torch.empty(len(data[0]), data[0].shape[1]), torch.empty(len(data[0]), data[0].shape[1])
    all_idx = set(range(len(data[0])))
    for i in range(len(data[0])):
        candidate_idx = list(all_idx-{i})
        idx = random.choice(candidate_idx)
        source[i] = data[0][i]
        target[i] = data[1][idx]
    return source, target


def _init_fn(worker_id):
    np.random.seed(3 + worker_id)
    
def test(model, test_dataset, num_workers=8, batch_size=128):
    gpu = False
    if torch.cuda.is_available():
        gpu = True
    device = torch.device("cuda" if gpu else "cpu")
    model.eval()
    model.to(device)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=_init_fn, shuffle=False)
    New_embs = torch.empty(len(test_dataset), model.num_seeds, model.out_dim)
    idx = 0
    for x, y in tqdm(test_dataloader):
        if gpu:
            x, y = x.cuda(), y.cuda()
        out = model(x)
        New_embs[idx:idx+x.shape[0], :, :] = out.detach().cpu()
        idx += x.shape[0]
    alignment_rest, hits, mr, mrr = greedy_alignment(np.array(New_embs[:,0,:].squeeze()), np.array(New_embs[:,1,:].squeeze()))
    return alignment_rest, hits, mr, mrr

def train(model, train_dataset, valid_dataset, storage_path, fold=1, epochs = 50, num_workers=8, batch_size=128, lr=0.0001):
    import copy
    gpu = False
    if torch.cuda.is_available():
        gpu = True
        print("\n##### GPU available! #####\n")
    device = torch.device("cuda" if gpu else "cpu")
    print()
    print("*"*60)
    print(f"FOLD {fold}: {model.name} starts training on {train_dataset.name}")
    print("*"*60)
    print()
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=_init_fn, shuffle=True)
    best_weights = copy.deepcopy(model.state_dict())
    best_hits1 = 0.0
    Accuracy_list = []
    t0 = time.time()
    for e in range(epochs):
        Acc = 0.0
        for x, y in tqdm(train_dataloader):
            if gpu:
                x, y = x.cuda(), y.cuda()
            out = model(x)
            loss = model.loss(out, y)
            Acc += model.score(out.detach(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Acc = (Acc/len(train_dataset)).item()
        print(f"Epoch {e+1}/{epochs} ... Loss: {loss.item()}, Acc: {Acc}")
        print("\n#### Validation ####")
        _, hits, _, _ = test(model, valid_dataset, num_workers, batch_size)
        print("#### Validation ####\n")
        Accuracy_list.append(Acc)
        if hits[0] > best_hits1:
            best_hits1 = hits[0]
            best_weights = copy.deepcopy(model.state_dict())
    t1 = time.time()
    duration = t1-t0
    model.load_state_dict(best_weights)
    torch.save(model, f"{storage_path}/SetTransformer_fold{fold}_{round(best_hits1, 2)}.pt")
    with open(f"{storage_path}/SetTransformer_fold{fold}_acc_list.json", "w") as file:
        json.dump({"train acc": Accuracy_list}, file)
    print("Best Hits1: ", best_hits1)
    return model, best_hits1, duration
    
