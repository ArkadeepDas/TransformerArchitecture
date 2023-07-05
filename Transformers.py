# In this code we are going to develope transformers from scratch

###################################################
####################### Encoder #######################
###################################################

# Steps to follow:
# 1) Input: Some source of text for machine translation
# 2) Input text -> Input embeddings.
# We add Positional Encoding with Input Embedding
# 3) Input embeddings to Multi-Head Attention
# Here Multi head attention is the main part
# Multi head attention takes 3 inputs: 1) values 2) keys 3) queries
# 4) Multi-Head Attention -> Normalization
# 5) Normalization -> Feed Forward Network
# 6) Feed Forward Network -> Normalization

# Here skip connection is also present
# This is a Transformer Encoder block

###################################################
####################### Decoder #######################
###################################################

# Steps to follow:
# 1) Output of transformer values and keys pass to decoder and queries comes from previous part of decode
# 2) Decoder have exactly same block of encoder block
# 3) Here queries is different input for decoder block
# 4) Output Embedding + Positional Encoding -> Musked Multi-Head Attention
# 5) Musked Multi-Head Attention + Normalization
# Skip connetion is present here
# 6) Transformer Block -> Linear layer
# 7) Linear layer -> Softmax

# This is Transformer Decoder block
# We can have multiple blocks of encoder and decoder

# Multi-Head Attention
# values -> linear layer, keys -> linear layers, queries -> linear layers
# Output of those -> Scaled dot-product -> Concat -> Linear Layer

import torch
import torch.nn as nn


# Bulid self attention class
# We have embeddings and we are going to split them into different parts
class SelfAttention(nn.Module):

    def __init__(self, embed_size, heads):
        super().__init__()
        # Assume embed_size = 256, it means a single word can be represented as 256 vector
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim *
            heads == embed_size), 'Embed size needs to be divisible by heads'

        # Assume Input size = 256 -> Output size = 256
        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, values, keys, queries):
        # No. of examples send in at the same time
        N = queries.shape[0]
        # Here this lengths are the sentence length as input
        value_len, key_len, query_len = values.shape[1], keys.shape[
            1], queries.shape[1]

        # Now let's apply linear transformer over queries, keys and values
        # Input embeddings goes through these linear layer to produce queries keys values
        # Assume 256 -> 256
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Split the embeddings into self.heads pieces
        # Basically we convert 256 -> 8x32 for values, keys and queries
        # Shape = (N, Sentence Len, 8, 32)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # We multiply metrix using einsum()
        # It helps to multiply several other dimentions
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # values shape: (N, value_len, heads, head_dim)
        # This is the formula of Self Attention
        # output of queries x keys = (N, heads, query_len, key_len)
        # We can imagine it as query_len: target sentence and key_len: source sentence
        # Concept is each word in target how much we pay attention to each word in our input

        # Flatten training examples with heads
        # We can think of it as matrix multiplication of 10x256 * 256x10 = 10x10
        # We assume 10 is sentence length and 256 is word embeddings
        energy = torch.einsum('nqhd, nkhd->nhqk', [queries, keys])

        # Now we apply Softmax activation and normalize accross key_len
        attention = torch.softmax(energy / (self.embed_size**(1 / 2)), dim=3)

        # Now multiply attention with values
        # It provides elements wise operations
        # Attention shape: (N, heads, query_len, key_len)
        # Values shape: (N, value_len, heads, head_dim)
        # Output after multiplication: (N, query_len, heads, head_dim)
        # key_len = value_len always. So we are multiply accross that dimention
        # k = key length, v = value lenght
        # Flatten last two dimention
        out = torch.einsum('nhqk, nvhd->nqhd', [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim)
        # After reshape we convert it to (N, 10, 256)
        # Assume Sentence length = 10, embed_size = 256

        # Now apply fully connected layer
        out = self.fc_out(out)

        return out


# Now we create Transformer Block
class TransformerBlock(nn.Module):

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        # We are going to use the self attention here
        self.attention = SelfAttention(embed_size=embed_size, heads=heads)
        # Then we apply normalization
        self.norm1 = nn.LayerNorm(embed_size)
        # Feed forward and Normalization layer
        # We are maping to some more nodes
        # In paper forward_expension = 4
        # After ReLU they map back to embed_size
        self.feef_forward = nn.Sequential(
            nn.Linear(in_features=embed_size,
                      out_features=forward_expansion * embed_size), nn.ReLU(),
            nn.Linear(in_features=forward_expansion * embed_size,
                      out_features=embed_size))
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries):
        # Adding attention layer
        attention = self.attention(values, keys, queries)
        # Adding skip connection
        skip = self.norm1(attention + queries)
        # Adding dropout
        dropout = self.dropout(skip)
        # Applying feed forward and Normalization
        forward = self.feef_forward(dropout)
        skip = self.norm2(forward + dropout)
        out = self.dropout(skip)

        return out


# Now both self attention and transformer block is done
# Now let's build the Encoder part
# Now we are going to do embedding
class Encoder(nn.Module):

    def __init__(self, source_vocab_size, embed_size, num_layers, heads,
                 device, forward_expension, dropout, max_length):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        # Word embedding
        # source_vocab_size = maximum number of unique token it can take
        # Here we are sending 10 for testing as sentence length
        self.word_embedding = nn.Embedding(source_vocab_size, embed_size)
        # Positional embedding
        # max_len = maximum number of unique oken it can take
        # Here we are sending 100 as position lenght. Actually both are same size.
        self.positional_embedding = nn.Embedding(max_length, embed_size)
        # If we want multiple transformer blocks then we can add here
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size=embed_size,
                             heads=heads,
                             dropout=dropout,
                             forward_expansion=forward_expension)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Now we initialize all of the things. Let's arrange them in forward()

    def forward(self, x):
        N, seq_length = x.shape
        # torch.arange() generates a sequence of numbers starting from the first argument and ending before the second argument
        # torch.expand() expand the tensors along with dimention
        # So we create a position vector of input text samples
        positions = torch.arange(0,
                                 seq_length).expand(N,
                                                    seq_length).to(self.device)
        # We pass input to embedding and add positional embedding
        out = self.word_embedding(x) + self.positional_embedding(positions)
        # The positional embedding help us to understand how words are positions
        out = self.dropout(out)

        # Here we only add one layer
        for layer in self.layers:
            out = layer(out, out, out)

        return out


# Now let's create the decoder block
# First buld the decoder block
class DecoderBlock(nn.Module):

    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super().__init__()
        # First Attention layer, then normalization layer, then transformer block
        self.attention = SelfAttention(embed_size=embed_size, heads=heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size=embed_size,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, values, keys):
        # Here x is input from our target
        # Values and keys are from encoder which is already run before this
        attention = self.attention(x, x, x)
        # Skip connection part
        norm = self.norm(attention + x)
        # We create the queries from target input and pass values, keys from encoder to transformer block
        queries = self.dropout(norm)
        out = self.transformer_block(values, keys, queries)

        return out


# Let's build the Decoder
class Decoder(nn.Module):

    def __init__(self, target_vocab_size, embed_size, num_layers, heads,
                 forward_expantion, dropout, max_len, device):
        super().__init__()
        self.device = device
        # Output word embeddings as input word embeddings
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        # Positional vector for output as input
        self.position_embedding = nn.Embedding(max_len, embed_size)

        # Multiple Decoder block
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expantion, dropout, device)
            for _ in range(num_layers)
        ])
        # target_vocab_size is the vocabulary size of the target vocabulary
        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    # This is the main understanding
    # One by one we marge together for decoder
    def forward(self, x, encoder_out):
        N, seq_length = x.shape
        # Set the position as encoder
        positions = torch.arange(0,
                                 seq_length).expand(N,
                                                    seq_length).to(self.device)
        # Now apply word embeddings and position embeddings
        x = self.word_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        # Now apply the Decoder
        for layer in self.layers:
            # Quries from decoder and values and keys are from encoder out
            x = layer(x, encoder_out, encoder_out)

        out = self.fc_out(x)
        return out


# Now let's put this together to understand all the steps one by one
class Transformer(nn.Module):

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=256,
                 num_layers=6,
                 forward_expention=4,
                 heads=8,
                 dropout=0,
                 device='cuda',
                 max_length=100):
        super().__init__()
        # Initializing Encoder and Decoder
        self.encoder = Encoder(source_vocab_size=src_vocab_size,
                               embed_size=embed_size,
                               num_layers=num_layers,
                               heads=heads,
                               device=device,
                               forward_expension=forward_expention,
                               dropout=dropout,
                               max_length=max_length)
        self.decoder = Decoder(target_vocab_size=trg_vocab_size,
                               embed_size=embed_size,
                               num_layers=num_layers,
                               heads=heads,
                               forward_expantion=forward_expention,
                               dropout=dropout,
                               device=device,
                               max_len=max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # Now combining every steps one by one
    def forward(self, src, trg):
        # Now let's pass the data to the model
        enc_src = self.encoder(src)
        out = self.decoder(trg, enc_src)
        return out


# Let's test the model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Input data of different shape
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7,
                                                    2]]).to(device)
    # Target data of different shape
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6,
                                                   2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 50
    trg_vocab_size = 50
    model = Transformer(src_vocab_size,
                        trg_vocab_size,
                        src_pad_idx,
                        trg_pad_idx,
                        device=device).to(device)
    out = model(x, trg)
    print("Output Shape: ")
    print(out.shape)