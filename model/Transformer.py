import torch
import tokenizers

import torch.nn as nn

from .Encoder import Encoder
from .Decoder import Decoder
from .PositionalEncoding import SineCosinePositionalEncoding

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, in_tokenizer: tokenizers.Tokenizer, out_tokenizer: tokenizers.Tokenizer,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 d_model: int = 512, num_heads: int = 8, max_tokens: int = 2048, device: torch.device = torch.device("cpu"),
                 special_tokens: dict ={"unknown": "[UNK]", "pad": "[PAD]", "start": "[CLS]", "end": "[SEP]"}, **kwargs):
        super(EncoderDecoderTransformer, self).__init__()

        # Store device
        self.device = device
        # Store params
        self.in_tokenizer = in_tokenizer
        self.out_tokenizer = out_tokenizer
        self.in_vocab_size = in_tokenizer.get_vocab_size()
        self.out_vocab_size = out_tokenizer.get_vocab_size()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_tokens = max_tokens
        self.special_tokens = special_tokens        

        # Generate embbeding layer
        self.in_embedding = nn.Embedding(self.in_vocab_size, d_model, device=device)
        self.out_embedding = nn.Embedding(self.out_vocab_size, d_model, device=device)
        # Generate positional encoding 
        self.pos_enc = SineCosinePositionalEncoding(d_model, max_tokens=max_tokens, device=device)

        # Generate encoder and decoder
        self.encoder = Encoder(num_encoder_layers, device=device, **kwargs)
        self.decoder = Decoder(num_decoder_layers, device=device, **kwargs)

        # Generate output and softmax layers
        self.linear = nn.Linear(self.d_model, self.out_vocab_size, device=device)
        self.softmax = nn.Softmax(dim=-1)

    def encode(self, x: torch.Tensor):
        # Generate mask
        enc_mask = self.encoder.generate_mask(x, self.in_tokenizer.token_to_id(self.special_tokens["pad"])).to(self.device)
        # Embedd inputs
        x = self.in_embedding(x)
        # Add positional encoding
        x = self.pos_enc(x)
        # Encoder pass
        return self.encoder(x, enc_mask), enc_mask
    
    def decode(self, x: torch.Tensor, y: torch.Tensor, enc_mask: torch.Tensor):
        # Generate mask
        dec_mask = self.decoder.generate_mask(y, self.out_tokenizer.token_to_id(self.special_tokens["pad"])).to(self.device)
        # Embedd inputs
        y = self.out_embedding(y)
        # Add positional encoding
        y = self.pos_enc(y)
        # Decoder pass
        return self.decoder(y, x, enc_mask, dec_mask)
    
    def generate(self, x: torch.Tensor, max_gen_tokens: int = 512):

        # Tokenize input
        #x = torch.tensor(self.in_tokenizer.encode(x).ids, dtype=torch.int32).unsqueeze(0).to(self.device)

        # Encode input
        x, enc_mask = self.encode(x)
        #end_token = self.out_tokenizer.token_to_id(self.special_tokens["end"])
        # Create initial output
        y = torch.empty((x.shape[0], 1), dtype=torch.int32, device=self.device)
        y[:, 0] = self.out_tokenizer.token_to_id(self.special_tokens["start"])

        # Start generative loop
        for i in range(1, max_gen_tokens):
            # Decode current state
            pred = self.decode(x, y, enc_mask)
            # Select only the last token
            pred = self.linear(pred[:, -1])
            # Apply softmax
            pred = self.softmax(pred)
            # Get the most probable token
            pred = torch.argmax(pred, dim=-1).unsqueeze(1)
            # Add new token
            y = torch.cat((y, pred), dim=1)


        return y



    def forward(self, x: torch.Tensor, y: torch.Tensor):

        # Encode input
        x, enc_mask = self.encode(x)
        
        # Decode output
        y = self.decode(x, y, enc_mask)

        # Linear output
        y = self.linear(y)

        return y
    
    