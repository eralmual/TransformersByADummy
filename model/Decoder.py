import torch
import torch.nn as nn
from .FeedForwardNetwork import FFN
from .MultiHeadAttention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, device: torch.device = torch.device("cpu")):
        super(DecoderLayer, self).__init__()
        # Store device
        self.device = device
        # Store params
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Create masked MHA layer
        self.masked_mha = MultiHeadAttention(self.d_model, self.num_heads, self.device)
        # Create encoder sublayers
        self.enc_mha = MultiHeadAttention(self.d_model, self.num_heads, self.device)
        self.enc_ffn = FFN(self.d_model, self.d_ff, self.device)
        # Create layer norm layers
        self.ln_mha = nn.LayerNorm(self.d_model, device=self.device)        
        self.ln_enc_mha = nn.LayerNorm(self.d_model, device=self.device)
        self.ln_ffn = nn.LayerNorm(self.d_model, device=self.device)
        

    def forward(self, y, enc_in, enc_mask=None, dec_mask=None):
        # Apply masked MHA layer
        y = self.ln_mha(y + self.masked_mha(y, y, y, dec_mask))
        # Apply encoder sublayers
        y = self.ln_enc_mha(y + self.enc_mha(y, enc_in, enc_in, enc_mask))
        y = self.ln_ffn(y + self.enc_ffn(y))

        return y
    

class Decoder(nn.Module):
    def __init__(self, num_layers: int = 6, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, device: torch.device = torch.device("cpu")):
        super(Decoder, self).__init__()
        # Store device
        self.device = device
        # Store params
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Generate encoder layers
        self.dec_layers = []
        for _ in range(self.num_layers):
            self.dec_layers.append(DecoderLayer(self.d_model, self.num_heads, self.d_ff, self.device))
        self.dec_layers = nn.ModuleList(self.dec_layers)

    def generate_mask(self, y: torch.Tensor, pad_id: int):
        # Generate decoder mask
        pad_mask = (y != pad_id).unsqueeze(1).unsqueeze(2)
        progress_mask = torch.tril(torch.ones(y.shape[1], y.shape[1])).type(torch.bool).to(self.device)
        return pad_mask & progress_mask


    def forward(self, y: torch.Tensor, enc_in: torch.Tensor, enc_mask: torch.Tensor = None, dec_mask: torch.Tensor = None):
        # Pass the input through all decoding layers
        for decoder_layer in self.dec_layers:
            y = decoder_layer(y, enc_in, enc_mask, dec_mask)

        return y