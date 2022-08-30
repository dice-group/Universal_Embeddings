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
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to use to load training data')
parser.add_argument('--input_size', type=int, default=300, help='Input size (embedding dimension)')
parser.add_argument('--proj_dim', type=int, default=128, help='Size of hidden layers')
parser.add_argument('--num_inds', type=int, default=32, help='Number of induced instances')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--num_seeds', type=int, default=1, help='Number of seed components in the output')
parser.add_argument('--ln', type=str2bool, default=False, help='Whether to use layer normalization')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
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
    test_res = {1: None, 2: None, 3: None, 4: None, 5: None}
    for fold in range(1,6):
        emb1, emb2, test_set, links = get_data(data_path[dataset], fold)
        S, T, S_test, T_test = get_source_and_target_matrices(links, emb1, emb2, given_test_set=test_set)
        S, T, S_test, T_test = torch.FloatTensor(S), torch.FloatTensor(T), torch.FloatTensor(S_test), torch.FloatTensor(T_test)
        input_size = S.shape[1]
        args.input_size = input_size
        corrupt_source, corrupt_target = generate_false_matching([S,T])
        labels = torch.cat([torch.ones(corrupt_source.shape[0]), -1*torch.ones(corrupt_source.shape[0])], 0).to(torch.long)
        source, target = torch.cat([S, corrupt_source], 0), torch.cat([T, corrupt_target], 0)
        data = [source, target, labels]
        train_dataset = AlignDataSet(data, dataset.upper())
        model = SetTransformer(args)
        model = train(model, train_dataset, data_path[dataset], fold, args.epochs, args.num_workers, args.batch_size, args.lr)
        if args.test:
            test_labels = torch.ones(S_test.shape[0])
            data_test = [S_test, T_test, test_labels]
            test_dataset = AlignDataSet(data_test, dataset.upper())
            alignment_rest, hits, mr, mrr = test(model, test_dataset)
            test_res[fold] = [list(hits), mr, mrr]
    with open(f"{data_path[dataset]}/SetTransformer_test_results.json", "w") as file:
        json.dump({"test results": test_res}, file)
        
