import torch
import math
import torch.nn

class InputEmbeddings(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x): 
        return self.embedding(x) * math.sqrt(self.d_model)
class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape 
        p = torch.zeros(seq_len, d_model)
        #Create a vector of shape(Seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model,2).float() * (-math.log(10000.0)/d_model))
        #Apply the sin to even and cosine to odd 
        p[:, 0::2] = torch.sin(position*div_term)
        p[:, 1::2] = torch.cos(position*div_term)

        p = p.unsqueeze(0)

        self.register_buffer('pe',pe)
    def forward(self, x): 
        x += (self.p[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x) 
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6): 
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim =-1, keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.bias
class FeedFordwardBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        # Batch, seq_len, d_model --> batch, seq_len, d_ff --> batch, seq_len, d_model
        return self.linear_2(torch.relu(self.dropout(self.linear_1(x))))
class MultiHeadedAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model%h==0, "d_model is not divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        #batch, h ,seq_len, d_k --> batch, h, seq_len, seq_len
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if Mask is not None: 
            attention_scores.masked_fill_(mask ==0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # batch, h, seq_len, seq_len
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # batch, seq_len, d_model --> batch, seq_len, d_model
        key = self.w_k(k) # batch, seq_len, d_model --> batch, seq_len, d_model
        value = self.w_v(v) # batch, seq_len, d_model --> batch, seq_len, d_model

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0, -1, self.h * self.d_k])
        #batch, seq_len, d_model --> batch, seq_len, d_model
        return self.w_o(x)
class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for i in range(2)])
    def forward(self, x, src_mask): 
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connections[1](s. self.feed_forward_block)
        return x
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self. norm = LayerNormalization()
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)









    

    
    
    
    



        


