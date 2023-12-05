import torch
import torch.nn as nn
from .FeedForwardNetwork import FFN
from .MultiHeadAttention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, device: torch.device = torch.device("cpu")):
        super(EncoderLayer, self).__init__()
        # Store device
        self.device = device
        # Store params
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Create Sub-Layers
        self.mha = MultiHeadAttention(self.d_model, self.num_heads, self.device)
        self.ffn = FFN(self.d_model, self.d_ff, self.device)
        # Create layer norm for MHA and FFN
        self.ln_mha = nn.LayerNorm(self.d_model, device=self.device)
        self.ln_ffn = nn.LayerNorm(self.d_model, device=self.device)

    def forward(self, x, mask=None):
        # Apply MHA sublayer
        x = self.ln_mha(x + self.mha(x, x, x, mask))
        # Apply FFN sublayer
        x = self.ln_ffn(x + self.ffn(x))

        return x
    
class Encoder(nn.Module):
    def __init__(self, num_layers: int = 6, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, device: torch.device = torch.device("cpu")):
        super(Encoder, self).__init__()
        # Store device
        self.device = device
        # Store params
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Generate encoder layers
        self.enc_layers = []
        for _ in range(self.num_layers):
            self.enc_layers.append(EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.device))
        self.enc_layers = nn.ModuleList(self.enc_layers)

    def generate_mask(self, x: torch.Tensor, pad_id: int):
        # Generate encoder mask
        return (x != pad_id).type(torch.bool).unsqueeze(1).unsqueeze(2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Pass the input through all encoding layers
        for encoder_layer in self.enc_layers:
            x = encoder_layer(x, mask)

        return x
