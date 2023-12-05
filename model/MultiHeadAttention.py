import math
import torch
import unittest
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    # We asume d_q = d_k = d_v = d_model / num_heads, as the paper suggested
    def __init__(self, d_model: int = 512, num_heads: int = 8, device: torch.device = torch.device("cpu")):
        super(MultiHeadAttention, self).__init__()
        # Must be multiples
        assert d_model % num_heads == 0
        # Store device
        self.device = device
        # Store parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        # Linear layers for QKV and output
        self.linear_q = nn.Linear(self.d_model, self.d_model, device=self.device)
        self.linear_k = nn.Linear(self.d_model, self.d_model, device=self.device)
        self.linear_v = nn.Linear(self.d_model, self.d_model, device=self.device)
        self.linear_out = nn.Linear(self.d_model, self.d_model, device=self.device)

        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        # We asumme imputs have shape (batch_size, num_tokens, d_model)
        # Linear projection on QKV
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # Reshape Q, K, V for multi-head attention
        # After this reshaping, the attention mechanism processes each head independently and in 
        # parallel, which is a key feature of the multi-head attention mechanism. 
        # Shape: (batch_size, num_heads, sequence_length, d_k)
        q = q.view(q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(v.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (batch_size, num_heads, d_k, sequence_length)
        k = k.view(k.size(0), -1, self.num_heads, self.d_k).transpose(1, 2).transpose(2, 3)

        # Scaled dot-product attention
        # Q * K, permute so (bs, h, sql, dk)*  (bs, h, dk, sql) = (bs, h, sql, sql)
        score = torch.matmul(q, k)
        # Scale by sqrt(dk)
        score /= self.sqrt_dk 
        
        # Apply mask
        if(mask is not None):
            score = score.masked_fill(mask == 0, -1e10)

        # Softmax
        score = self.softmax(score)
        # Score * V, (bs, h, sql, sql) * (bs, h, sql, dk) = (bs, h, sql, dk)
        score = torch.matmul(score, v)

        # Transpose from (batch_size, num_heads, sequence_length, d_k) to (batch_size, sequence_length, num_heads, d_k)
        score = score.transpose(1, 2)
        # Reshape as output takes input dims: d_k * num_heads = d_model 
        #  Reshape as (batch_size, sequence_length, num_heads * d_k)
        score = score.contiguous().view(score.size(0), -1, self.num_heads * self.d_k)
        # Output scale
        score = self.linear_out(score)

        return score  
    
class TestMultiHeadAttention(unittest.TestCase):
    def test_forward(self):
        # Set seed for reproducibility
        torch.manual_seed(42)

        # Define input dimensions
        batch_size = 16
        sequence_length = 10
        dim_keys = 64
        num_heads = 8
        dim_model = dim_keys

        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create instance of MultiHeadAttention
        multihead_attention = MultiHeadAttention(dim_model, num_heads, device=dev)

        # Generate random input tensors
        q = torch.randn((batch_size, sequence_length, dim_keys))
        k = torch.randn((batch_size, sequence_length, dim_keys))
        v = torch.randn((batch_size, sequence_length, dim_keys))

        # Forward pass through the MultiHeadAttention layer
        output = multihead_attention(q, k, v)

        # Ensure the output tensor has the correct shape
        expected_shape = (batch_size, sequence_length, dim_keys)
        self.assertEqual(output.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()