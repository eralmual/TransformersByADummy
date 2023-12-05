import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, device: torch.device = torch.device("cpu")):
        super(FFN, self).__init__()
        # Store device
        self.device = device
        # Store params
        self.d_model = d_model
        self.d_ff = d_ff

        # Generate FF layers
        self.linear_in = nn.Linear(self.d_model, self.d_ff, bias=True, device=self.device)
        self.linear_out = nn.Linear(self.d_ff, self.d_model, bias=True, device=self.device)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.linear_out(x)
        
        return x