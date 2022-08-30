import torch
from torch.nn.utils.rnn import pad_sequence

class AlignDataSet(torch.utils.data.Dataset):
    def __init__(self, data: list, name: str):
        self.Source = data[0]
        self.Target = data[1]
        self.Label = data[2]
        self.name = name
        super(AlignDataSet, self).__init__()
        
    def __len__(self):
        return len(self.Source)
    
    def __getitem__(self, idx):
        source = self.Source[idx]
        target = self.Target[idx]
        return pad_sequence([source, target], batch_first=True, padding_value=0), self.Label[idx]