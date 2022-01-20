# Split into train, validation data

import argparse
from sklearn.model_selection import train_test_split

def get_entities(triples):
    entities = set()
    for l in triples:
        e1, _, e2 = l.split("\t")
        entities.update({e1, e2})
    return entities

def get_relations(triples):
    relations = set()
    for l in triples:
        relations.add(l.split('\t')[1])
    return relations

def filter_triples(triples, train, valid_entities, valid_relations):
    remaining, removed = [], []
    for l in triples:
        e1, r, e2 = l.split('\t')
        if e1 in valid_entities and e2 in valid_entities and r in valid_relations:
            remaining.append(l)
        else:
            removed.append(l)
    return train+removed, remaining

def write_to_file(storage_path, data):
    with open(storage_path, 'w') as file:
        file.writelines(data)

def split_data(kg, test_size):
    """
    Function for splitting intial set of triples into train and test in a way that all entities and relations in the test data also appear in the training data
    
    -- args: (1) knowledge graph name, should coincide with the folder name. Also make sure that the folder contains the knowledge graph in a file triples.txt,
             (2) test split size
    
    -- instructions: creates train.txt and test.txt
    
    -- returns: None
    """
    print()
    print(f'Creating train and test data for {kg}...')
    print()
    path = kg+'/triples.txt'
    with open(path, 'r') as file:
        data = file.readlines()
#        data = list(filter(lambda x: len(x.split('\t')) == 3, data))
        print(len(data))
        
    train, temp_test = train_test_split(data, test_size=test_size, random_state=142)
    train_entities = get_entities(train)
    train_relations = get_relations(train)
    
    train, test = filter_triples(temp_test, train, train_entities, train_relations)
    
    print(f'\nStatistics train: {100*float(len(train))/len(data)}%,\
          test: {100*float(len(test))/len(data)}%')
    storage_paths = [kg+'/train.txt', kg+'/test.txt']
    for path, d in zip(storage_paths, [train, test]):
        write_to_file(path, d)
    print('\n Done saving data')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--kgs', type=str, nargs='+', default=['DBpedia'], help='Sequence of knowledge graphs to preprocess, this reduces to creating creating train.txt and test.txt')
    parser.add_argument('--test_size', type=float, default=0.15, help='Test size during split')
    
    args = parser.parse_args()
    
    for kg in args.kgs:
        split_data(kg, args.test_size)
    
    