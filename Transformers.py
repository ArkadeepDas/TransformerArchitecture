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

    def forward(self, values, keys, queries, mask):
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

        if mask is not None:
            # If a element mask value = 0 then we want to shut them off. So we give them very less value which have 0
            energy = energy.masked_fill(mask == 0, float('-1e20'))

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