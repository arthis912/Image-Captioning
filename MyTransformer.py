from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class MyTransformer(nn.Module):
    def __init__(self, patch_size, num_classes, emb_dim, encoder_depth, decoder_depth, nhead, mlp_dim, dropout, tgt_vocab_size):
        super(MyTransformer, self).__init__()
        self.vision_transformer = ViT(
            image_size=256,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=emb_dim,
            depth=encoder_depth,
            heads=nhead,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=dropout
        )

        self.vision_encoder = Extractor(self.vision_transformer)

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=emb_dim, 
                                       nhead=nhead,
                                       dim_feedforward=mlp_dim, 
                                       dropout=dropout),
            num_layers=decoder_depth
        )
        self.generator = nn.Linear(emb_dim, tgt_vocab_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_dim)
        self.positional_encoding = PositionalEncoding(
            emb_dim, dropout=dropout)


    @property
    def device(self):
        return next(self.parameters()).device
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt, PAD_IDX):
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)

        tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
        return tgt_mask, tgt_padding_mask

    def forward(self,
                src: Tensor,
                trg: Tensor,
                PAD_IDX: Tensor):

        _, memory = self.vision_encoder(src)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        tgt_mask, tgt_padding_mask = self.create_mask(src, trg, PAD_IDX)

        logits = self.decode(tgt_emb, memory.transpose(0, 1), tgt_mask, tgt_padding_mask)

        return self.generator(logits)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_key_padding_mask: Tensor):
        return self.transformer_decoder(
            tgt, 
            memory, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask
        )


