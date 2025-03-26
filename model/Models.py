import jsonlines
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ESIMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(ESIMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True)
        self.inference_encoder = nn.LSTM(hidden_size * 8, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 8, 3)  # 3分类
        self.act = nn.Tanh()

    def forward(self, premise, hypothesis):
        a_emb = self.embedding(premise)  # (batch_size, seq_len, emb_dim)
        b_emb = self.embedding(hypothesis)
        a_encoded, _ = self.encoder(a_emb)  # (batch_size, seq_len, 2*hidden_size)
        b_encoded, _ = self.encoder(b_emb)

        attn = torch.matmul(a_encoded, b_encoded.transpose(1, 2))  # (batch_size, a_len, b_len)
        a_tilde = torch.matmul(torch.softmax(attn, dim=2), b_encoded)
        b_tilde = torch.matmul(torch.softmax(attn, dim=1).transpose(1, 2), a_encoded)

        # presentation reinforcement
        m_a = torch.cat([a_encoded, a_tilde, a_encoded - a_tilde, a_encoded * a_tilde], dim=2)
        m_b = torch.cat([b_encoded, b_tilde, b_encoded - b_tilde, b_encoded * b_tilde], dim=2)

        # inference compose
        v_a, _ = self.inference_encoder(m_a)
        v_b, _ = self.inference_encoder(m_b)
        v_a = torch.cat([v_a.max(dim=1)[0], v_a.mean(dim=1)], dim=1)
        v_b = torch.cat([v_b.max(dim=1)[0], v_b.mean(dim=1)], dim=1)
        v = torch.cat([v_a, v_b], dim=1)

        # classifier
        logits = self.act(self.fc(v))
        return logits


class DecomposableAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(DecomposableAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.F = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU()
        )
        self.G = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Linear(2*hidden_size, 3)

    def forward(self, premise, hypothesis):
        a_emb = self.embedding(premise)  # (batch_size, seq_len, emb_dim)
        b_emb = self.embedding(hypothesis)

        a_bar = self.F(a_emb)  # (batch_size, seq_len, hidden_size)
        b_bar = self.F(b_emb)

        attn = torch.matmul(a_bar, b_bar.transpose(1, 2))  # (batch_size, seq_len, seq_len)

        beta = torch.matmul(torch.softmax(attn, dim=2), b_bar)  # (batch_size, seq_len, hidden_size)
        alpha = torch.matmul(torch.softmax(attn, dim=1).transpose(1, 2), a_bar)

        v_1 = self.G(torch.concat([a_bar, beta], dim=2))  # (batch_size, seq_len, hidden_size)
        v_2 = self.G(torch.concat([b_bar, alpha], dim=2))
        v_1 = torch.sum(v_1, dim=1)  # (batch_size, 1, hidden_size)
        v_2 = torch.sum(v_2, dim=1)

        logits = self.classifier(torch.concat([v_1, v_2], dim=1))

        return logits




