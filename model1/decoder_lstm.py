import torch
import torch.nn as nn

from .attention import Attention

# Attention Decoder
# starter code  from https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch/execution


class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim,
                 decoder_dim, drop_prob):
        super().__init__()
        # save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(
            embed_size+encoder_dim, decoder_dim, bias=True)
        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions):
        device = features.device
        # (batch_size, num_captions, embed_dim)
        embeds = self.embedding(captions)
        # (batch_size, decoder_dim)
        h, c = self.init_hidden_state(features)          
        seq_length = len(captions[0])-1
        batch_size = captions.size(0)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        for s in range(seq_length):
            _, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            preds[:, s] = output
        return preds

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def predict(self, features, max_len, start_caption, end_caption, pad_caption):        
        device = features.device
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)
        words = torch.tensor(start_caption).tile((batch_size, 1)).to(device)
        embeds = self.embedding(words)
        captions = torch.tensor(pad_caption).tile(
            (batch_size, max_len)).to(device)        
        is_ended = torch.tensor(False).repeat(batch_size).to(device)

        alphas = torch.zeros(batch_size, max_len, features.shape[1]).to(device)
        outputs = torch.zeros(batch_size, max_len, self.vocab_size).to(device)
        for i in range(max_len):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            preds = output.argmax(dim=1)

            alphas[:, i] = alpha
            outputs[:, i] = output
            captions[:, i] = torch.where(is_ended, pad_caption, preds)
            
            is_ended |= (preds == end_caption)
            if torch.all(is_ended):
                break
            embeds = self.embedding(preds.unsqueeze(1))
        return captions, outputs, alphas
