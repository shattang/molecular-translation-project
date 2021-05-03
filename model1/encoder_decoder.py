
import torch.nn as nn

from .encoder_cnn import EncoderCNN
from .decoder_lstm import DecoderLSTM

# starter code from https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch/execution


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim,
                 encoder_dim, decoder_dim, drop_prob, fine_tune,
                 resnet_model=None):
        super().__init__()
        self.encoder = EncoderCNN(resnet_model=resnet_model, fine_tune=fine_tune)
        self.decoder = DecoderLSTM(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            drop_prob=drop_prob
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def predict(self, images, max_len, start_caption, end_caption, pad_caption):
        features = self.encoder(images)
        outputs = self.decoder.predict(
            features, max_len, start_caption, end_caption, pad_caption)
        return outputs
