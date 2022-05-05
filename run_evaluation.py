
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import time, gc
from sklearn.neighbors import NearestNeighbors
import random
from tqdm import tqdm
gc.enable()
random.seed(42)

def get_source_and_target_matrices(alignment_dict, entity2vec1, entity2vec2, given_test_set=None, emb_dim=200, test_size=0.1, scale=True, shift=True):
    """This function takes the dictionary of aligned entities between two KGs and their corresponding embeddings (as entity to vector dictionaries)
    and returns S, T, S_eval, T_eval, and R defined as follows:
    
    -- S: Normalized and scaled large subset of the source embeddings, i.e. the matrix of aligned entity embeddings in the first knowledge graph
    
    -- T: Normalized and scaled large subset of the matrix of aligned entity embeddings in the second knowledge graph
    
    -- S_eval and T_eval are portions of S and T sampled for evaluation if test_size > 0
    
    -- R: The rotation matrix that most closely maps S to T, i.e. ||A@S-T|| is minimized
    
    The mean and standard deviation of S, T are also returned
    """
    t0 = time.time()
    if test_size > 0:
        if given_test_set is None:
            train_ents, eval_ents = train_test_split(list(alignment_dict.keys()), test_size=test_size, random_state=42)
        else:
            eval_ents = given_test_set
            train_ents = list(set(alignment_dict.keys())-set(eval_ents))
    else:
        train_ents = alignment_dict.keys()
    
    S = np.empty((len(train_ents), emb_dim))
    T = np.empty((len(train_ents), emb_dim))
    if test_size > 0:
        S_eval = np.empty((len(eval_ents), emb_dim))
        T_eval = np.empty((len(eval_ents), emb_dim))

    for i, key in tqdm(enumerate(train_ents), total=len(train_ents), desc='Computing S and T'):
        S[i] = entity2vec1[key] if isinstance(entity2vec1, dict) else entity2vec1.loc[key].values
        T[i] = entity2vec2[alignment_dict[key]] if isinstance(entity2vec2, dict) else entity2vec2.loc[alignment_dict[key]].values
        
    if test_size > 0:
        for i, key in tqdm(enumerate(eval_ents), total=len(eval_ents), desc='Computing S_eval and T_eval'):
            S_eval[i] = entity2vec1[key] if isinstance(entity2vec1, dict) else entity2vec1.loc[key].values
            T_eval[i] = entity2vec2[alignment_dict[key]] if isinstance(entity2vec2, dict) else entity2vec2.loc[alignment_dict[key]].values
        
    print('\nNow computing R...')
    # Center and scale data
    mean_S = S.mean(axis=0)
    mean_T = T.mean(axis=0)
    scale_S = np.sqrt(((S-mean_S)**2).sum()/S.shape[0]) # scale, see https://en.wikipedia.org/wiki/Procrustes_analysis
    scale_T = np.sqrt(((T-mean_T)**2).sum()/T.shape[0])
    print('Scale S: ', scale_S)
    
    if shift and scale:
        R, loss = orthogonal_procrustes((S-mean_S)/scale_S, (T-mean_T)/scale_T, check_finite=True)
        print('\nCompleted after '+str(time.time()-t0)+' seconds')
    elif shift:
        R, loss = orthogonal_procrustes((S-mean_S), (T-mean_T), check_finite=True)
        print('\nCompleted after '+str(time.time()-t0)+' seconds')
    elif scale:
        R, loss = orthogonal_procrustes(S/scale_S, T/scale_T, check_finite=True)
        print('\nCompleted after '+str(time.time()-t0)+' seconds')
    else:
        R, loss = orthogonal_procrustes(S, T, check_finite=True)
        print('\nCompleted after '+str(time.time()-t0)+' seconds')
        
    print('Alignment loss: ', loss)
    if test_size > 0:
        if shift and scale:
            return scale_S, scale_T, mean_S, mean_T, (S-mean_S)/scale_S, (T-mean_T)/scale_T, (S_eval-mean_S)/scale_S, (T_eval-mean_T)/scale_T, R
        elif shift:
            return scale_S, scale_T, mean_S, mean_T, S-mean_S, T-mean_T, S_eval-mean_S, T_eval-mean_T, R
        elif scale:
            return scale_S, scale_T, mean_S, mean_T, S/scale_S, T/scale_T, S_eval/scale_S, T_eval/scale_T, R
        else:
            return scale_S, scale_T, mean_S, mean_T, S, T, S_eval, T_eval, R
    else:
        if shift and scale:
            return scale_S, scale_T, mean_S, mean_T, (S-mean_S)/scale_S, (T-mean_T)/scale_T, R
        elif shift:
            return scale_S, scale_T, mean_S, mean_T, S-mean_S, T-mean_T, R
        elif scale:
            return scale_S, scale_T, mean_S, mean_T, S/scale_S, T/scale_T, R
        else:
            return scale_S, scale_T, mean_S, mean_T, S, T, R

def get_non_aligned_entity_embedding_matrices(alignment_dict, entity2vec1, entity2vec2, scale_S, scale_T, mean_S, mean_T, emb_dim=200):
    """
    Inputs the dictionary of aligned entities between two KGs and their corresponding embeddings, and returns the normalized embedding matrices of 
    
    non-aligned entities
    """
    A_neg_S = np.empty((len(entity2vec1)-len(alignment_dict), emb_dim))
    keys = sorted(set(entity2vec1.keys() if isinstance(entity2vec1, dict) else entity2vec1.index)-set(alignment_dict.keys()))
    for i, key in tqdm(enumerate(keys), total=A_neg_S.shape[0], desc='Computing A_neg_S...'):
        A_neg_S[i] = entity2vec1[key] if isinstance(entity2vec1, dict) else entity2vec1.loc[key].values
    
    B_neg_T = np.empty((len(entity2vec2)-len(alignment_dict), emb_dim))
    keys = sorted(set(entity2vec2.keys() if isinstance(entity2vec2, dict) else entity2vec2.index)-set(alignment_dict.values()))
    for i, key in tqdm(enumerate(keys), total=B_neg_T.shape[0], desc='Computing B_neg_T...'):
        B_neg_T[i] = entity2vec2[key] if isinstance(entity2vec2, dict) else entity2vec2.loc[key].values
        
    return (A_neg_S-mean_S)/scale_S, (B_neg_T-mean_T)/scale_T
    

def evaluate_alignment_knn(S_eval, T_eval, R, assume_known=True, hit_values = [1, 3, 10]):
    """The function takes the evaluation sets, i.e. correct alignments that were left out, and returns the hits@ and MRR results w.r.t. correct alignments
    
    --assume_known. A boolean variable. When set to True, the alignment results are computed using the fact that the test links are known
    
    """
    print('#'*50)
    print('Evaluation started...')
    print('#'*50)
    model = NearestNeighbors(n_neighbors=S_eval.shape[0], n_jobs=-1)
    print('Fitting 1...')
    model.fit(T_eval)
    print('Predicting 1...')
    if assume_known:
        preds = model.kneighbors((S_eval@R+T_eval)/2, n_neighbors=S_eval.shape[0], return_distance=False)
    else:
        preds = model.kneighbors(S_eval, n_neighbors=S_eval.shape[0], return_distance=False)
    Hits1 = np.zeros(len(hit_values))
    MRR1 = 0.0
    for i in tqdm(range(S_eval.shape[0]), total=S_eval.shape[0]):
        pred_idx = (preds[i]==i).nonzero()[0][0] # if i in preds[i] else S_eval.shape[0]
        MRR1 += (1./(pred_idx+1))
        for j in range(len(Hits1)):
            if pred_idx < hit_values[j]:
                Hits1[j] += 1.0/S_eval.shape[0]
    MRR1 = MRR1/S_eval.shape[0]
    
    model = NearestNeighbors(n_neighbors=S_eval.shape[0], n_jobs=-1)
    print('\nFitting 2...')
    if assume_known:
        model.fit((S_eval@R+T_eval)/2)
    else:
        model.fit(S_eval)
    print('Predicting 2...')
    preds = model.kneighbors(T_eval, n_neighbors=S_eval.shape[0], return_distance=False)
    Hits2 = np.zeros(len(hit_values))
    MRR2 = 0.0
    for i in tqdm(range(S_eval.shape[0]), total=S_eval.shape[0]):
        pred_idx = (preds[i]==i).nonzero()[0][0] # if i in preds[i] else S_eval.shape[0]
        MRR2 += (1./(pred_idx+1))
        for j in range(len(Hits2)):
            if pred_idx < hit_values[j]:
                Hits2[j] += 1.0/S_eval.shape[0]
    MRR2 = MRR2/S_eval.shape[0]
    
    Hits = (Hits1+Hits2)/2
    MRR = (MRR1+MRR2)/2
    print()
    print(', '.join([f'Hits@{hit_values[it]}: {Hits[it]}' for it in range(len(Hits))]+[f'MRR: {MRR}']))


import argparse

path_embeddings = ['Shallom_EnFr_15K_V1/Shallom_entity_embeddings.csv', 'Shallom_EnFr_15K_V2/Shallom_entity_embeddings.csv',
                  'Shallom_EnFr_100K_V1/Shallom_entity_embeddings.csv', 'Shallom_EnFr_100K_V1/Shallom_entity_embeddings.csv',
                  'Experiments/EN_DE_15K_V1/Shallom_entity_embeddings.csv', 'Experiments/EN_DE_15K_V2/Shallom_entity_embeddings.csv',
                  'Experiments/EN_DE_100K_V1/Shallom_entity_embeddings.csv', 'Experiments/EN_DE_100K_V1/Shallom_entity_embeddings.csv']
path_ent_links = ['OpenEA_dataset_v1.1/EN_FR_15K_V1/ent_links', 'OpenEA_dataset_v1.1/EN_FR_15K_V2/ent_links',
                 'OpenEA_dataset_v1.1/EN_FR_100K_V1/ent_links', 'OpenEA_dataset_v1.1/EN_FR_100K_V2/ent_links',
                 'OpenEA_dataset_v1.1/EN_DE_15K_V1/ent_links', 'OpenEA_dataset_v1.1/EN_DE_15K_V2/ent_links',
                 'OpenEA_dataset_v1.1/EN_DE_100K_V1/ent_links', 'OpenEA_dataset_v1.1/EN_DE_100K_V2/ent_links']
path_test_links = ['OpenEA_dataset_v1.1/EN_FR_15K_V1/721_5fold/1/test_links', 'OpenEA_dataset_v1.1/EN_FR_15K_V2/721_5fold/1/test_links',
                 'OpenEA_dataset_v1.1/EN_FR_100K_V1/721_5fold/1/test_links', 'OpenEA_dataset_v1.1/EN_FR_100K_V2/721_5fold/1/test_links',
                 'OpenEA_dataset_v1.1/EN_DE_15K_V1/721_5fold/1/test_links', 'OpenEA_dataset_v1.1/EN_DE_15K_V2/721_5fold/1/test_links',
                 'OpenEA_dataset_v1.1/EN_DE_100K_V1/721_5fold/1/test_links', 'OpenEA_dataset_v1.1/EN_DE_100K_V2/721_5fold/1/test_links']
names = ['EnFr_15K_V1', 'EnFr_15K_V2', 'EnFr_100K_V1', 'EnFr_100K_V2', 'EnDe_15K_V1', 'EnDe_15K_V2', 'EnDe_100K_V1', 'EnDe_100K_V2']

parser = argparse.ArgumentParser()
parser.add_argument('--path_embeddings', nargs='+', type=str, default=path_embeddings, help='Path to entity embeddings of paired KGs')
parser.add_argument('--path_ent_links', nargs='+', type=str, default=path_ent_links, help='Path to entity links')
parser.add_argument('--path_test_links', nargs='+', type=str, default=path_test_links, help='Path to test links')
parser.add_argument('--emb_dim', type=int, default=300, help='KG embedding dimension')
args=parser.parse_args()

for path_emb, path_ent_link, path_test, name in zip(args.path_embeddings, args.path_ent_links, args.path_test_links, names):
    print('#'*50)
    print(f'Evaluation on {name}...')
    print('#'*50)
    pair_kg_emb = pd.read_csv(path_emb)
    if 'Fr' in name:
        kg2_emb = pair_kg_emb[pair_kg_emb['Unnamed: 0'].apply(lambda x: 'fr.dbpedia.org' in x)]
    else:
        kg2_emb = pair_kg_emb[pair_kg_emb['Unnamed: 0'].apply(lambda x: 'de.dbpedia.org' in x)]

    kg1_emb = pair_kg_emb.iloc[np.setdiff1d(np.arange(pair_kg_emb.shape[0]),\
                                                                np.array(kg2_emb.index))].set_index('Unnamed: 0')

    kg2_emb = kg2_emb.set_index('Unnamed: 0')
    assert kg2_emb.shape[1] == args.emb_dim, f'Expecting emb_dim to be equal to the pretrained embedding dimension but got {kg2_emb.shape[1]} and {args.emb_dim}'

    with open(path_ent_link) as file:
        kg1_to_kg2_ents = file.read().strip().split('\n')
    kg1_to_kg2_ents = dict([line.split('\t') for line in kg1_to_kg2_ents])

    with open(path_test) as file:
        test_set = file.read().strip().split('\n')
    test_set = [line.split('\t')[0] for line in test_set]

    print('Test size: ', len(test_set))
    _, _, _, _, _, _, S_eval, T_eval, R = get_source_and_target_matrices(kg1_to_kg2_ents,\
                                                    kg1_emb, kg2_emb, given_test_set=test_set, emb_dim=args.emb_dim, test_size=0.2)


    hit_values = [1, 5, 10]
    evaluate_alignment_knn(S_eval, T_eval, R, assume_known=True, hit_values=hit_values)

