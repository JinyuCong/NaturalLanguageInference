import jsonlines
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, raw_data, word_2_index, seq_len):
        self.raw_data = raw_data
        self.seq_len = seq_len
        self.word_2_index = word_2_index
        self.index_2_word = {index: word for word, index in self.word_2_index.items()}
        self.label_mapping = {"neutral": 0, "entailment": 1, "contradiction": 2}
        self.data = self._build_corpus()

    def _build_corpus(self):
        data = []
        for premise, hypothesis, label in self.raw_data:
            premise, hypothesis, label = premise.lower(), hypothesis.lower(), label.lower()

            premise_index = [self.word_2_index.get(word_premise, 1) for word_premise in premise.split()[:self.seq_len]]
            premise_index += [0 for _ in range(self.seq_len - len(premise_index))]

            hypothesis_index = [self.word_2_index.get(word_hypothesis, 1) for word_hypothesis in hypothesis.split()[:self.seq_len]]
            hypothesis_index += [0 for _ in range(self.seq_len - len(hypothesis_index))]

            label_index = self.label_mapping[label]
            data.append((premise_index, hypothesis_index, label_index))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise_index = self.data[idx][0]
        hypothesis_index = self.data[idx][1]
        label_index = self.data[idx][2]
        return torch.Tensor(premise_index).to(torch.int), torch.Tensor(hypothesis_index).to(torch.int), label_index


class ESIMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
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

        # 增强表示
        m_a = torch.cat([a_encoded, a_tilde, a_encoded - a_tilde, a_encoded * a_tilde], dim=2)
        m_b = torch.cat([b_encoded, b_tilde, b_encoded - b_tilde, b_encoded * b_tilde], dim=2)

        # 推理组合
        v_a, _ = self.inference_encoder(m_a)
        v_b, _ = self.inference_encoder(m_b)
        v_a = torch.cat([v_a.max(dim=1)[0], v_a.mean(dim=1)], dim=1)
        v_b = torch.cat([v_b.max(dim=1)[0], v_b.mean(dim=1)], dim=1)
        v = torch.cat([v_a, v_b], dim=1)

        # 预测
        logits = self.act(self.fc(v))
        return logits