import torch
import torch.nn as nn

class SineCosinePositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_tokens=2048, device=torch.device("cpu")):
        super(SineCosinePositionalEncoding, self).__init__()
        
        # Generate the positional encoding vector
        x = torch.arange(max_tokens, device=device).unsqueeze(1) / torch.pow(10000, (2 * torch.arange(d_model, device=device)) / d_model)
        self.pos_enc = torch.zeros(size=(max_tokens, d_model), device=device)
        self.pos_enc[:, 0::2] = torch.sin(x[:, 0::2])
        self.pos_enc[:, 1::2] = torch.cos(x[:, 1::2])

    def forward(self, x):
        return x + self.pos_enc[:x.shape[1], :]