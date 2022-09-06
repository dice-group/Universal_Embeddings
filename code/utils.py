import pandas as pd
import numpy as np
import torch
import json
from evaluation import greedy_alignment
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_source_and_target_matrices(alignment_dict, entity2vec1, entity2vec2, train_ents, valid_ents, test_ents, return_test_embs=True):
    """This function takes the dictionary of aligned entities between two KGs and their corresponding embeddings (as entity to vector dictionaries)
    and returns S, T, S_valid, T_valid, S_test, and T_test:
    
    -- S: Subset of the source embeddings, i.e. the matrix of aligned entity embeddings in the first knowledge graph (for training)
    
    -- T: Subset of the matrix of aligned entity embeddings in the second knowledge graph (for training)
    
    -- S_valid and T_valid are the embedding matrices corresponding to SameAs links in the validation set (10% of the whole data)
    
    -- S_test and T_test are the embedding matrices corresponding to SameAs links in the test set (70% of the whole data)
    
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
        if np.abs(S).mean(0).max() < 0.1: # For numerical stability
            S, T = 100*S, 100*T
            S_test, T_test = 100*S_test, 100*T_test
            S_valid, T_valid = 100*S_valid, 100*T_valid
        return S, T, S_valid, T_valid, S_test, T_test
    else:
        if np.abs(S).mean(0).max() < 0.1: # For numerical stability
            S, T = 100*S, 100*T
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
    rand_idx = list(range(data[0].shape[0]))
    random.shuffle(rand_idx)
    source = data[0]
    target = data[1][rand_idx]
    return source, target


def _init_fn(worker_id):
    np.random.seed(3 + worker_id)
    
def test(model, test_dataset, num_workers=8, batch_size=128):
    model.eval()
    model.to(device)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=_init_fn, shuffle=False)
    New_embs = torch.empty(len(test_dataset), 2, model.out_dim)
    idx = 0
    for x, _ in tqdm(test_dataloader):
        x = x.to(device)
        out = model(x)
        New_embs[idx:idx+x.shape[0], 0, :] = out[0].detach().cpu()
        New_embs[idx:idx+x.shape[0], 1, :] = out[1].detach().cpu()
        idx += x.shape[0]
    alignment_rest, hits, mr, mrr = greedy_alignment(np.array(New_embs[:,0,:]), np.array(New_embs[:,1,:]))
    return alignment_rest, hits, mr, mrr

def create_mult_negatives(i, x, y, n=None):
    if n is None:
        n = x.shape[0] // 2
    x_pos = x[i,:,:].unsqueeze(0).to(device)
    y_pos = y[i].unsqueeze(0).to(device)
    x_neg = torch.cat([x[i,0,:].repeat(n,1).unsqueeze(1).to(device), x.index_select(0,torch.tensor(random.sample(\
                            [j for j in range(x.shape[0]) if j!=i], n)))[:,1,:].unsqueeze(1).to(device)], 1)
    return torch.cat([x_pos, x_neg], 0), torch.cat([y_pos, -1*torch.ones(n).to(device)], 0)

def get_batch(x, y, n=32):
    features = []
    labels = []
    for i in range(x.shape[0]):
        xx, yy = create_mult_negatives(i, x, y, n=n)
        features.append(xx)
        labels.append(yy)
    return torch.cat(features, 0), torch.cat(labels, 0).long()

def train(model, train_dataset, valid_dataset, storage_path, n=32, fold=1, epochs = 50, num_workers=8, batch_size=128, lr=0.0001):
    import copy
    if torch.cuda.is_available():
        print("\n##### GPU available! #####\n")
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
    #Accuracy_list = []
    t0 = time.time()
    for e in range(epochs):
        #Acc = 0.0
        Loss = 0.0
        for x, y in tqdm(train_dataloader):
            n = min(n, x.shape[0])
            xx, yy = get_batch(x,y,n)
            out = model(xx)
            loss = model.loss(out, yy)
            #Acc += model.score(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss += loss.item()
        #Acc = (Acc/len(train_dataset)).item()
        #print(f"Epoch {e+1}/{epochs} ... Loss: {loss.item()}, Acc: {Acc}")
        #print("\n#### Validation ####")
        print(f"#### Epoch {e+1}/{epochs} ... Loss {Loss}")
        _, hits, _, _ = test(model, valid_dataset, num_workers, batch_size)
        print("#### Validation ####\n")
        #Accuracy_list.append(Acc)
        if hits[0] > best_hits1:
            best_hits1 = hits[0]
            best_weights = copy.deepcopy(model.state_dict())
    t1 = time.time()
    duration = t1-t0
    model.load_state_dict(best_weights)
    torch.save(model, f"{storage_path}/SetTransformer_fold{fold}.pt")
    #with open(f"{storage_path}/SetTransformer_fold{fold}_acc_list.json", "w") as file:
    #    json.dump({"train acc": Accuracy_list}, file)
    print("Best Hits1: ", best_hits1)
    return model, best_hits1, duration