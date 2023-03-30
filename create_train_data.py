import os
from tqdm import tqdm
def read_write(kg_name):
    if not os.path.exists(f'OpenEA_dataset_v2.0/{kg_name}/KG1'):
        os.mkdir(f'OpenEA_dataset_v2.0/{kg_name}/KG1')
    if not os.path.exists(f'OpenEA_dataset_v2.0/{kg_name}/KG2'):
        os.mkdir(f'OpenEA_dataset_v2.0/{kg_name}/KG2')
    data1 = []
    data2 = []
    for d_type in ['attr_triples_1', 'rel_triples_1']:
        with open(f'OpenEA_dataset_v2.0/{kg_name}/{d_type}') as file:
            data1.extend(file.read().split('\n'))
    for d_type in ['attr_triples_2', 'rel_triples_2']:
        with open(f'OpenEA_dataset_v2.0/{kg_name}/{d_type}') as file:
            data2.extend(file.read().split('\n'))
    with open(f'OpenEA_dataset_v2.0/{kg_name}/KG1/train.txt', 'w') as file:
        for triple in data1:
            file.write(triple+'\n')
    with open(f'OpenEA_dataset_v2.0/{kg_name}/KG1/test.txt', 'w') as file:
        for triple in data1[-500:]:
            file.write(triple+'\n')
    with open(f'OpenEA_dataset_v2.0/{kg_name}/KG2/train.txt', 'w') as file:
        for triple in data2:
            file.write(triple+'\n')
    with open(f'OpenEA_dataset_v2.0/{kg_name}/KG2/test.txt', 'w') as file:
        for triple in data2[-500:]:
            file.write(triple+'\n')
        
for kg_name in tqdm(os.listdir('OpenEA_dataset_v2.0'), desc='looping through datasets...'):
    read_write(kg_name)