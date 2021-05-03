import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Bahdanau Attention
# starter code from https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch/execution


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.attention_dim = attention_dim
        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)
        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        u_hs = self.U(features)  # (batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state)  # (batch_size,attention_dim)
        # (batch_size,num_layers,attemtion_dim)
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))
        attention_scores = self.A(combined_states)  # (batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(
            2)  # (batch_size,num_layers)
        alpha = F.softmax(attention_scores, dim=1)  # (batch_size,num_layers)
        # (batch_size,num_layers,features_dim)
        attention_weights = features * alpha.unsqueeze(2)
        attention_weights = attention_weights.sum(
            dim=1)  # (batch_size,num_layers)
        return alpha, attention_weights
