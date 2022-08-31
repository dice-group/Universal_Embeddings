import torch, numpy as np
from torch.utils.data import DataLoader
import argparse
import random
from model import SetTransformer
from utils import get_source_and_target_matrices, get_data, generate_false_matching, train, test
from dataloader import AlignDataSet
import json

seed = 3
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['t', 'true', 'y', 'yes', '1']:
        return True
    elif v.lower() in ['f', 'false', 'n', 'no', '0']:
        return False
    else:
        raise ValueError('Invalid boolean value.')
        
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, nargs='+', default=['dbpenfr15kv1', 'dbpenfr15kv2',\
                                                                'dbpenfr100kv1', 'dbpenfr100kv2',
                                                                'dbpende15kv1', 'dbpende15kv2',\
                                                                'dbpende100kv1', 'dbpende100kv2'],
                                                                help='Name of aligned KGs')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to use to load data as into batches')
parser.add_argument('--chunk_size', type=int, default=128, help='Size of chunks along the embedding vector of each entity')
parser.add_argument('--input_size', type=int, default=300, help='Input size (embedding dimension)')
parser.add_argument('--proj_dim', type=int, default=256, help='Size of hidden layers')
parser.add_argument('--num_inds', type=int, default=32, help='Number of induced components')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--precision', type=float, default=0.2, help='The precision or confidence for the alignment')
parser.add_argument('--margin', type=float, default=0.0, help='The margin in CosineEmbeddingLoss')
parser.add_argument('--num_seeds', type=int, default=2, help='Number of seed components in the output')
parser.add_argument('--ln', type=str2bool, default=False, help='Whether to use layer normalization')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
parser.add_argument('--test', type=str2bool, default=True, help='Whether to run evaluation on the test data')
parser.add_argument('--final', type=str2bool, default=False, help='Align the given KGs (training and test data are combined) for final universal embeddings')
args = parser.parse_args()

data_path = {"dbpenfr15kv1": "../EN_FR_15K_V1" , "dbpenfr15kv2": "../EN_FR_15K_V2",\
             "dbpenfr100kv1": "../EN_FR_100K_V1", "dbpenfr100kv2": "../EN_FR_100K_V2",\
             "dbpende15kv1": "../EN_DE_15K_V1" , "dbpende15kv2": "../EN_DE_15K_V2",\
             "dbpende100kv1": "../EN_DE_100K_V1", "dbpende100kv2": "../EN_DE_100K_V2"
                }

for dataset in args.datasets:
    test_res = {i: None for i in range(1, args.folds+1)}
    valid_res = {i: None for i in range(1, args.folds+1)}
    for fold in range(1,args.folds+1):
        emb1, emb2, train_ents, valid_ents, test_ents, links = get_data(data_path[dataset], fold)
        S, T, S_valid, T_valid, S_test, T_test = get_source_and_target_matrices(links, emb1, emb2, train_ents, valid_ents, test_ents)
        S, T,  = torch.FloatTensor(S), torch.FloatTensor(T)
        S_valid, T_valid = torch.FloatTensor(S_valid), torch.FloatTensor(T_valid)
        S_test, T_test = torch.FloatTensor(S_test), torch.FloatTensor(T_test)
        input_size = S.shape[1]
        args.input_size = input_size
        corrupt_source, corrupt_target = generate_false_matching([S,T])
        labels = torch.cat([torch.ones(corrupt_source.shape[0]), -1*torch.ones(corrupt_source.shape[0])], 0).to(torch.long)
        source, target = torch.cat([S, corrupt_source], 0), torch.cat([T, corrupt_target], 0)
        data = [source, target, labels]
        train_dataset = AlignDataSet(data, dataset.upper(), args.chunk_size)
        model = SetTransformer(args)
        model = train(model, train_dataset, data_path[dataset], fold, args.epochs, args.num_workers, args.batch_size, args.lr)
        
        ## Validation
        print("\n ### Validation... ###")
        valid_labels = torch.ones(S_valid.shape[0])
        data_valid = [S_valid, T_valid, valid_labels]
        valid_dataset = AlignDataSet(data_valid, dataset.upper(), args.chunk_size)
        alignment_rest, hits, mr, mrr = test(model, valid_dataset, args.num_workers, args.batch_size)
        valid_res[fold] = [list(hits), mr, mrr]
        print()
            
        ## Test
        
        if args.test:
            print("\n ### Test... ###")
            test_labels = torch.ones(S_test.shape[0])
            data_test = [S_test, T_test, test_labels]
            test_dataset = AlignDataSet(data_test, dataset.upper(), args.chunk_size)
            alignment_rest, hits, mr, mrr = test(model, test_dataset, args.num_workers, args.batch_size)
            test_res[fold] = [list(hits), mr, mrr]
            print()
            
    with open(f"{data_path[dataset]}/SetTransformer_test_results.json", "w") as file:
        json.dump({"test results": test_res}, file)
        
    with open(f"{data_path[dataset]}/SetTransformer_valid_results.json", "w") as file:
        json.dump({"validation results": valid_res}, file)
