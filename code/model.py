from modules import *

class SetTransformer(nn.Module):
    def __init__(self, kwargs):
        super(SetTransformer, self).__init__()
        self.name = 'SetTransformer'
        
        self.enc = nn.Sequential(ISAB(kwargs.input_size, kwargs.proj_dim, kwargs.num_heads, kwargs.num_inds, ln=kwargs.ln),
                                 ISAB(kwargs.proj_dim, kwargs.proj_dim, kwargs.num_heads, kwargs.num_inds, ln=kwargs.ln),
                                 ISAB(kwargs.proj_dim, kwargs.output_size, kwargs.num_heads, kwargs.num_inds, ln=kwargs.ln))
        
        
        self.dec = nn.Sequential(PMA(kwargs.output_size, kwargs.num_heads, kwargs.num_seeds, ln=kwargs.ln),
                                 SAB(kwargs.output_size, kwargs.output_size, kwargs.num_heads, ln=kwargs.ln))
        
        self.Loss = nn.CosineEmbeddingLoss(margin=kwargs.margin)
        self.similarity = nn.CosineSimilarity()
        self.out_dim = kwargs.output_size
        self.num_seeds = kwargs.num_seeds
        self.precision = kwargs.precision
        
    def loss(self, out, y):
        for i in range(len(out)-1):
            if i == 0:
                loss = self.Loss(out[i],out[i+1], y)
            else:
                loss = loss + self.Loss(out[i],out[i+1], y)
        return loss
    
    def score(self, out, y):
        scores = torch.cat([self.similarity(out[i].detach(), out[i+1].detach()).unsqueeze(1) for i in range(len(out)-1)], 1).min(dim=1).values
        return torch.eq(torch.sign(scores-self.precision), y).sum().cpu()
        
        
    def forward(self, X):
        out = []
        for i in range(X.shape[1]):
            out.append(self.dec(self.enc(X[:,i,:].unsqueeze(1))).squeeze())
        return out
