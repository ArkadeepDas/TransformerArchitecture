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
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim *
            heads == embed_size), 'Embed size needs to be divisible by heads'

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, values, keys, queries, mask):
        # No. of examples send in at the same time
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[
            1], queries.shape[1]

        # Split the embeddings into self.heads pieces
        # Basically we convert 256 -> 8x32 for values, keys and queries
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # We multiply metrix using einsum()
        # It helps to multiply several other dimentions
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # values shape: (N, value_len, heads, head_dim)
        # output of queries x keys = (N, heads, query_len, key_len)
        # We can imagine it as query_len: target sentence and key_len: source sentence
        # Concept is each word in target how much we pay attention to each word in our input