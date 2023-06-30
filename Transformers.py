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
