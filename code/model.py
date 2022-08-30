from modules import *

class SetTransformer(nn.Module):
    def __init__(self, kwargs):
        super(SetTransformer, self).__init__()
        self.name = 'SetTransformer'
        self.enc = ISAB(kwargs.input_size, kwargs.proj_dim, kwargs.num_heads, kwargs.num_inds, ln=kwargs.ln)
        self.dec = SAB(kwargs.proj_dim, kwargs.proj_dim, kwargs.num_heads, ln=kwargs.ln)
        self.Loss = nn.CosineEmbeddingLoss()
        self.similarity = nn.CosineSimilarity()
        self.out_dim = kwargs.proj_dim
        
    def loss(self, out, y):
        for i in range(out.shape[1]-1):
            if i == 0:
                loss = self.Loss(out[:,0,:].squeeze(),out[:,1,:].squeeze(), y)
            else:
                loss = loss + self.Loss(out[:,i,:].squeeze(),out[:,i+1,:].squeeze(), y)
        return loss
    
    def score(self, out, y):
        scores = torch.cat([self.similarity(out[:,i,:].squeeze(), out[:,i+1,:].squeeze()).unsqueeze(1) for i in range(out.shape[1]-1)], 1).min(dim=1).values
        return (torch.sign(scores) == y).sum()
        
        
    def forward(self, X):
        enc_out = self.enc(X)
        return self.dec(enc_out)
