import torch
import torch.nn as nn
import torch.nn.functional as F


class ESIMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(ESIMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
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


class SelfAttentionEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(attn_out + x)


class ESIMModelWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads=4):
        super(ESIMModelWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = SelfAttentionEncoder(embedding_dim, num_heads)
        self.inference_encoder = SelfAttentionEncoder(embedding_dim * 4, num_heads)
        self.fc = nn.Linear(embedding_dim * 16, 3)
        self.act = nn.Tanh()

    def forward(self, premise, hypothesis):

        a_emb = self.embedding(premise)  # (batch_size, seq_len, emb_dim)
        b_emb = self.embedding(hypothesis)

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

        logits = self.act(self.fc(v))  # (hidden_size * 8, 3)

        return logits
