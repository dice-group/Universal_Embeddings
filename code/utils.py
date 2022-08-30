import pandas as pd
import numpy as np
import torch
import json
from evaluation import greedy_alignment
from tqdm import tqdm
import random
from torch.utils.data import DataLoader

def get_source_and_target_matrices(alignment_dict, entity2vec1, entity2vec2, given_test_set, return_test_set=True):
    """This function takes the dictionary of aligned entities between two KGs and their corresponding embeddings (as entity to vector dictionaries)
    and returns S, T, S_test, T_test, and R defined as follows:
    
    -- S: Subset of the source embeddings, i.e. the matrix of aligned entity embeddings in the first knowledge graph
    
    -- T: Subset of the matrix of aligned entity embeddings in the second knowledge graph
    
    -- S_test and T_test are the embedding matrices corresponding to the lef-out SameAs links
    
    """
    
    if return_test_set:
        assert given_test_set is not None, "given_test_set cannot be None if return_test_set is True"
        test_ents = given_test_set
        train_ents = list(set(alignment_dict.keys())-set(test_ents))
    else:
        train_ents = alignment_dict.keys()
    emb_dim = entity2vec1.shape[1]
    S = np.empty((len(train_ents), emb_dim))
    T = np.empty((len(train_ents), emb_dim))
    if return_test_set:
        S_test = np.empty((len(test_ents), emb_dim))
        T_test = np.empty((len(test_ents), emb_dim))

    for i, key in tqdm(enumerate(train_ents), total=len(train_ents), desc='Computing S and T'):
        S[i] = entity2vec1[key] if isinstance(entity2vec1, dict) else entity2vec1.loc[key].values
        T[i] = entity2vec2[alignment_dict[key]] if isinstance(entity2vec2, dict) else entity2vec2.loc[alignment_dict[key]].values
        
    if return_test_set:
        for i, key in tqdm(enumerate(test_ents), total=len(test_ents), desc='Computing S_test and T_test'):
            S_test[i] = entity2vec1[key] if isinstance(entity2vec1, dict) else entity2vec1.loc[key].values
            T_test[i] = entity2vec2[alignment_dict[key]] if isinstance(entity2vec2, dict) else entity2vec2.loc[alignment_dict[key]].values
    
    if return_test_set:
        return S, T, S_test, T_test
    else:
        return S, T
        
        
def get_data(dataset, fold):
    data = pd.read_csv(f"{dataset}/Shallom_entity_embeddings.csv")
    emb1 = data[data['Unnamed: 0'].apply(lambda x: 'http://dbpedia.org' in x)].set_index('Unnamed: 0')
    emb2 = data.iloc[np.setdiff1d(np.arange(data.shape[0]),np.array(emb1.index))].set_index('Unnamed: 0')

    with open(f"{dataset}/KG/ent_links") as file:
        kg1_kg2_links = file.read().strip().split('\n')
        kg1_kg2_links = dict([line.split('\t') for line in kg1_kg2_links])

    with open(f"{dataset}/KG/721_5fold/{fold}/test_links") as file:
        test_set = file.read().strip().split('\n')
    test_set = [line.split('\t')[0] for line in test_set]
    
    return emb1, emb2, test_set, kg1_kg2_links


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


def train(model, train_dataset, storage_path, fold=1, epochs = 50, num_workers=8, batch_size=128, lr=0.0001):
    import copy
    print()
    print("*"*30)
    print(f"{model.name} starts training on {train_dataset.name}")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    Accuracy_list = []
    for e in range(epochs):
        Acc = 0.0
        for x, y in tqdm(train_dataloader):
            out = model(x)
            loss_val = model.loss(out, y)
            Acc += model.score(out.detach(), y)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        Acc = (Acc/len(train_dataset)).item()
        print(f"Epoch {e+1}/{epochs} ... Loss: {loss_val.item()}, Acc: {Acc}")
        Accuracy_list.append(Acc)
        if Acc > best_acc:
            best_acc = Acc
            best_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    torch.save(model, f"{storage_path}/SetTransformer_fold{fold}_{round(best_acc, 2)}.pt")
    with open(f"{storage_path}/SetTransformer_fold{fold}_acc_list.json", "w") as file:
        json.dump({"train acc": Accuracy_list}, file)
    return model
    
def test(model, test_dataset, num_workers=8, batch_size=128):
    model.eval()
    Acc = 0.0
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    New_embs =torch.empty(len(test_dataloader), 2, model.out_dim)
    idx = 0
    for x, y in tqdm(test_dataloader):
        out = model(x)
        Acc += model.score(out.detach(), y)
    New_embs[idx:x.shape[0], :, :] = out.detach()
    idx += x.shape[0]
    print("Test acc: ", Acc/len(test_dataset))
    alignment_rest, hits, mr, mrr = greedy_alignment(np.array(New_embs[:,0,:].squeeze()), np.array(New_embs[:,1,:].squeeze()))
    return alignment_rest, hits, mr, mrr