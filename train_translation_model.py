import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy

print(torch.__version__)

# Embedding
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        #[123, 0, 23, 5] -> [[..512..], [...512...], ...]
        return self.embed(x)


#Positional encoding
# d_model: emb_size
class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, max_seq_len: int = 80):
        super().__init__()
        self.d_model = d_model

        #Create constant positional encoding matrix
        pos_matrix = torch.zeros(max_seq_len, d_model)

        # for pos in range(max_seq_len):
        #     for i in range(0, d_model, 2):
        #         pe_matrix[pos, i] = math.sin(pos/1000**(2*i/d_model))
        #         pe_matrix[pos, i+1] = math.cos(pos/1000**(2*i/d_model))
        #
        # pos_matrix = pe_matrix.unsqueeze(0) # Add one dimension for batch size

        den = torch.exp(-torch.arange(0, d_model, 2) * math.log(1000) / d_model)
        pos = torch.arange(0, max_seq_len).reshape(max_seq_len, 1)
        pos_matrix[:, 1::2] = torch.cos(pos * den)
        pos_matrix[:, 0::2] = torch.sin(pos * den)
        pos_matrix = pos_matrix.unsqueeze(0)


        self.register_buffer('pe', pos_matrix) #Register as persistent buffer

    def forward(self, x):
        # x is a sentence after embedding with dim (batch, number of words, vector dimension)
        seq_len = x.size()[1]
        x = x + self.pe[:, :seq_len]
        return x

## Scaled Dot-Product Attention layer
def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    # Shape of q and k are the same, both are (batch_size, seq_len, d_k)
    # Shape of v is (batch_size, seq_len, d_v)
    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1]) # size (bath_size, seq_len, d_k)

    # Apply mask to scores
    # <pad>
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, value=-1e9)

    # Softmax along the last dimension
    attention_weights = F.softmax(attention_scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    output = torch.matmul(attention_weights, v)
    return output

# Multi-Head Attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = self.d_v = d_model // n_heads

        # self attention linear layers
        #Linear layers for q, k, v vectors generation in different heads
        self.q_linear_layers = []
        self.k_linear_layers = []
        self.v_linear_layers = []
        for i in range(n_heads):
            self.q_linear_layers.append(nn.Linear(d_model, self.d_k))
            self.k_linear_layers.append(nn.Linear(d_model, self.d_k))
            self.v_linear_layers.append(nn.Linear(d_model, self.d_v))

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(n_heads*self.d_v, d_model)

    def forward(self, q, k, v, mask=None):
        multi_head_attention_outputs = []
        for q_linear, k_linear, v_linear in zip(self.q_linear_layers,
                                                self.k_linear_layers,
                                                self.v_linear_layers):
            new_q = q_linear(q) # size: (batch_size, seq_len, d_k)
            new_k = q_linear(k) # size: (batch_size, seq_len, d_k)
            new_v = q_linear(v) # size: (batch_size, seq_len, d_v)

            # Scaled Dot-Product attention
            head_v = scaled_dot_product_attention(new_q, new_k, new_v, mask, self.dropout) # (batch_size, seq_len,
            multi_head_attention_outputs.append(head_v)

        # Concat
        # import pdb; pdb.set_trace()
        concat = torch.cat(multi_head_attention_outputs, -1) # (batch_size, seq_len, n_heads*d_v)

        # Linear layer to recover to original shap
        output = self.out(concat) # (batch_size, seq_len, d_model)

        return output

# Feed Forward layer
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)

        return x

# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps= 1e-6):
        super().__init__()
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.beta = nn.Parameter(torch.zeros(self.d_model))
        self.eps = eps

    def forward(self, x):
        # x size: (batch_size, seq_len, d_model)
        x_hat = (x -x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps)
        x_tilde = self.alpha * x_hat + self.beta
        return x_tilde

#Encoder layers
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads: int, droput: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(n_heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(droput)
        self.dropout_2 = nn.Dropout(droput)

    def forward(self, x, mask):
        #import pdb; pdb.set_trace()
        x = x + self.dropout_1(self.multi_head_attention(x, x, x, mask))
        x = self.norm_1(x)

        x = x + self.dropout_2(self.feed_forward(x))
        x = self.norm_2(x)
        return x

#Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.norm_3 = LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.multi_head_attention_1 = MultiHeadAttention(n_heads, d_model)
        self.multi_head_attention_2 = MultiHeadAttention(n_heads, d_model)

        self.feed_forward = FeedForward(d_model)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        x = self.dropout_1(self.multi_head_attention_1(x, x, x, trg_mask))
        x = x + self.norm_1(x)

        x = self.dropout_2(self.multi_head_attention_2(x, encoder_output, encoder_output, src_mask))
        x = x + self.norm_2(x)

        x = self.dropout_3(self.feed_forward(x))
        x = x + self.norm_3(x)

        return x


def clone_layer(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Encoder & Decoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, n_heads):
        super().__init__()
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.encoder_layers = clone_layer(EncoderLayer(d_model, n_heads), N)
        self.norm = LayerNorm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for encoder in self.encoder_layers:
            x = encoder(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, n_heads):
        super().__init__()
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.decoder_layers = clone_layer(DecoderLayer(d_model, n_heads), N)
        self.norm = LayerNorm(d_model)

    def forward(self, trg, encoder_output, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for decoder in self.decoder_layers:
            x = decoder(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)

#Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, trg_vocab_size: int, d_model: int, N, n_heads: int):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, N, n_heads)
        self.decoder = Decoder(trg_vocab_size, d_model, N, n_heads)
        self.linear = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg, src_mask, trg_mask):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(trg, encoder_output, src_mask, trg_mask)
        output = self.linear(decoder_output)

        return output

#Data processing
import spacy
from torchtext.data.utils import get_tokenizer
from collections import Counter
import io
from torchtext.vocab import vocab

src_data_path = 'data/english.txt'
trg_data_path = 'data/french.txt'

en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

en_vocab = build_vocab(src_data_path, en_tokenizer)
fr_vocab = build_vocab(trg_data_path, fr_tokenizer)

def data_process(src_path, trg_path):
    raw_en_iter = iter(io.open(src_path, encoding="utf8"))
    raw_fr_iter = iter(io.open(trg_path, encoding="utf8"))
    data = []
    for (raw_en, raw_fr) in zip (raw_en_iter, raw_fr_iter):
        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)], dtype=torch.long)
        fr_tensor_ = torch.tensor([fr_vocab[token] for token in fr_tokenizer(raw_fr)], dtype= torch.long)
        data.append((en_tensor_, fr_tensor_))
    return data

train_data = data_process(src_data_path, trg_data_path)

#Train transformer
d_model= 512
n_heads = 8
N = 6
src_vocab_size = len(en_vocab.vocab)
trg_vocab_size = len(fr_vocab.vocab)

BATH_SIZE = 32
PAD_IDX = en_vocab['<pad>']
BOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def generate_batch(data_batch):
    en_batch, fr_batch = [], []
    for (en_item, fr_item) in data_batch:
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        fr_batch.append(torch.cat([torch.tensor([BOS_IDX]), fr_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    fr_batch = pad_sequence(fr_batch, padding_value=PAD_IDX)
    return en_batch, fr_batch

train_iter = DataLoader(train_data, batch_size=BATH_SIZE, shuffle=True, collate_fn=generate_batch)

model = Transformer(src_vocab_size, trg_vocab_size, d_model, N, n_heads)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)



import numpy as np

def create_mask(src_input, trg_input):
    # Source input mask

    pad = en_vocab['<pad>']
    src_mask = (src_input != pad).unsqueeze(1)

    #Target input mask
    trg_mask = (trg_input != pad).unsqueeze(1)

    seq_len = trg_input.size(1)
    nopeak_mask = np.tril(np.ones((1, seq_len, seq_len)), k=0).astype('uint8')
    nopeak_mask = torch.from_numpy(nopeak_mask) != 0
    trg_mask = trg_mask & nopeak_mask

    return src_mask, trg_mask

import time


def train_model(n_epochs, output_interval=100):
    model.train()
    start = time.time()

    for epoch in range(n_epochs):

        total_loss = 0
        for i, batch in enumerate(train_iter):

            src_input = batch[0].transpose(0, 1)  # size (batch_size, seq_len)
            trg = batch[1].transpose(0, 1)  # size (batch_size, seq_len)

            trg_input = trg[:, :-1]
            ys = trg[:, 1:].contiguous().view(-1)

            # create src & trg masks
            src_mask, trg_mask = create_mask(src_input, trg_input)
            preds = model(src_input, trg_input, src_mask, trg_mask)

            optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=1)
            loss.backward()
            optimizer.step()

            total_loss += loss.data

            if (i + 1) % output_interval == 0:
                avg_loss = total_loss / output_interval
                print('time = {}, epoch = {}, iter = {}, loss = {}'.format((time.time() - start) / 60,
                                                                           epoch + 1,
                                                                           i + 1,
                                                                           avg_loss))
                total_loss = 0
                start = time.time()


train_model(3, output_interval=1)