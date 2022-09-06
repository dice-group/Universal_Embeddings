import torch
from torch.nn.utils.rnn import pad_sequence

class AlignDataSet(torch.utils.data.Dataset):
    def __init__(self, data: list, name: str, chunk_size=10):
        self.features = [data[i] for i in range(len(data)-1)]
        self.Label = data[-1]
        self.name = name
        self.chunk_size = chunk_size
        super(AlignDataSet, self).__init__()
        
    def __len__(self):
        return len(self.Label)
    
    def __getitem__(self, idx):
        features = []
        for i in range(len(self.features)):
            features.append(torch.FloatTensor(self.features[i][idx]))
        #if self.chunk_size < source.shape[0]:
        #    source = torch.cat([chunk.unsqueeze(0) for chunk in pad_sequence(torch.split(source, self.chunk_size), batch_first=True)], 0)
        #    target = torch.cat([chunk.unsqueeze(0) for chunk in pad_sequence(torch.split(target, self.chunk_size), batch_first=True)], 0)
        #print("target: ", target.shape)
        # .reshape(-1, self.chunk_size)
        return pad_sequence(features, batch_first=True), self.Label[idx]