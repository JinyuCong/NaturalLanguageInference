import torch
import torch.nn as nn


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
    def __init__(self, vocab_size, embedding_dim, projected_dim, hidden_size, glove_embedding_matrix):
        super(DecomposableAttentionModel, self).__init__()
        # use the pretrained GloVe embedding matrix and freeze the learning
        self.embedding = nn.Embedding.from_pretrained(glove_embedding_matrix, freeze=True)
        # project the embedding matrix from dimension 300 to dimension 200
        self.projection = nn.Linear(embedding_dim, projected_dim)

        self.F = nn.Sequential(
            nn.Linear(projected_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.G = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(2*hidden_size, 3)

    def forward(self, premise, hypothesis):
        a_emb = self.embedding(premise)  # (batch_size, seq_len, emb_dim)
        b_emb = self.embedding(hypothesis)

        a_proj = self.projection(a_emb)  # (batch_size, seq_len, proj_dim)
        b_proj = self.projection(b_emb)

        a_bar = self.F(a_proj)  # (batch_size, seq_len, hidden_size)
        b_bar = self.F(b_proj)

        attn = torch.matmul(a_bar, b_bar.transpose(1, 2))  # (batch_size, seq_len, seq_len)

        beta = torch.matmul(torch.softmax(attn, dim=2), b_bar)  # (batch_size, seq_len, hidden_size)
        alpha = torch.matmul(torch.softmax(attn, dim=1).transpose(1, 2), a_bar)

        v_1 = self.G(torch.concat([a_bar, beta], dim=2))  # (batch_size, seq_len, hidden_size)
        v_2 = self.G(torch.concat([b_bar, alpha], dim=2))
        v_1 = torch.sum(v_1, dim=1)  # (batch_size, 1, hidden_size)
        v_2 = torch.sum(v_2, dim=1)

        # classifier
        logits = self.classifier(torch.concat([v_1, v_2], dim=1))
        return logits


'''
if __name__ == "__main__":
    # 假设参数
    vocab_size = 10000
    embed_dim = 300
    hidden_size = 100
    num_classes = 3  # 蕴含、中立、矛盾

    # 初始化模型（可加载预训练词向量）
    model = LSTMAttentionEntailment(vocab_size, embed_dim, hidden_size, num_classes)

    # 模拟输入
    batch_size = 32
    premise = torch.randint(0, vocab_size, (batch_size, 20))  # 假设前提长度20
    hypothesis = torch.randint(0, vocab_size, (batch_size, 15))  # 假设假设长度15

    # 前向传播
    logits = model(premise, hypothesis)
    print(logits.shape)  # 输出: [32, 3]
'''