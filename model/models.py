import torch
import torch.nn as nn
import torch.nn.functional as F


class ESIMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, pretrained_embeddings=None):
        super(ESIMModel, self).__init__()
        self.attention_map = None
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True)
        self.inference_encoder = nn.LSTM(hidden_size * 8, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 8, 3)
        self.act = nn.Tanh()

    def forward(self, premise, hypothesis):

        a_emb = self.embedding(premise)  # (batch_size, seq_len, emb_dim)
        b_emb = self.embedding(hypothesis)

        a_encoded, _ = self.encoder(a_emb)  # (batch_size, seq_len, 2*hidden_size)
        b_encoded, _ = self.encoder(b_emb)

        attn = torch.matmul(a_encoded, b_encoded.transpose(1, 2))  # (batch_size, a_len, b_len)

        self.attention_map = F.softmax(attn, dim=-1)  # save the attention map for visualisation

        a_tilde = torch.matmul(F.softmax(attn, dim=2), b_encoded)
        b_tilde = torch.matmul(F.softmax(attn, dim=1).transpose(1, 2), a_encoded)

        m_a = torch.cat([a_encoded, a_tilde, a_encoded - a_tilde, a_encoded * a_tilde], dim=2)
        m_b = torch.cat([b_encoded, b_tilde, b_encoded - b_tilde, b_encoded * b_tilde], dim=2)

        v_a, _ = self.inference_encoder(m_a)
        v_b, _ = self.inference_encoder(m_b)

        v_a = torch.cat([v_a.max(dim=1)[0], v_a.mean(dim=1)], dim=1)
        v_b = torch.cat([v_b.max(dim=1)[0], v_b.mean(dim=1)], dim=1)
        v = torch.cat([v_a, v_b], dim=1)

        logits = self.act(self.fc(v))  # (hidden_size * 8, 3)

        return logits


# ESIM model LSTM replacement with multihead attention
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros((seq_len, d_model))
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SelfAttentionEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, embedding_dim),
            nn.Dropout(0.3)
        )
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        output = self.ln2(x + ff_out)
        return output


class ESIMModelWithAttention(nn.Module):
    def __init__(self, vocab_size, seq_len, embedding_dim, num_heads=4, pretrained_embeddings=None):
        super(ESIMModelWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False
        self.pe = PositionalEncoding(seq_len, embedding_dim)
        self.encoder = nn.Sequential(
            SelfAttentionEncoder(embedding_dim, num_heads, ff_hidden=256),
            SelfAttentionEncoder(embedding_dim, num_heads, ff_hidden=256),
            SelfAttentionEncoder(embedding_dim, num_heads, ff_hidden=256)
        )
        self.inference_encoder = SelfAttentionEncoder(embedding_dim * 4, num_heads, ff_hidden=256)
        self.fc = nn.Linear(embedding_dim * 16, 3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, premise, hypothesis):

        a_emb = self.embedding(premise)  # (batch_size, seq_len, emb_dim)
        b_emb = self.embedding(hypothesis)

        a_emb = self.pe(a_emb)  # (batch_size, seq_len, emb_dim)
        b_emb = self.pe(b_emb)

        a_encoded = self.encoder(a_emb)  # (batch_size, seq_len, 2*hidden_size)
        b_encoded = self.encoder(b_emb)

        attn = torch.matmul(a_encoded, b_encoded.transpose(1, 2))  # (batch_size, a_len, b_len)

        a_tilde = torch.matmul(F.softmax(attn, dim=2), b_encoded)
        b_tilde = torch.matmul(F.softmax(attn, dim=1).transpose(1, 2), a_encoded)

        m_a = torch.cat([a_encoded, a_tilde, a_encoded - a_tilde, a_encoded * a_tilde], dim=2)
        m_b = torch.cat([b_encoded, b_tilde, b_encoded - b_tilde, b_encoded * b_tilde], dim=2)

        v_a = self.inference_encoder(m_a)
        v_b = self.inference_encoder(m_b)

        v_a = torch.cat([v_a.max(dim=1)[0], v_a.mean(dim=1)], dim=1)
        v_b = torch.cat([v_b.max(dim=1)[0], v_b.mean(dim=1)], dim=1)
        v = torch.cat([v_a, v_b], dim=1)

        logits = self.dropout(self.fc(v))  # (hidden_size * 8, 3)

        return logits
