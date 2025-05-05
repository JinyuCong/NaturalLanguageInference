import torch
import torch.nn as nn
import torch.nn.functional as F


class ESIMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(ESIMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, bidirectional=True)
        self.inference_encoder = nn.LSTM(hidden_size * 8, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 8, 3)  # 3分类
        self.act = nn.Tanh()

    def forward(self, premise, premise_mask, hypothesis, hypothesis_mask):
        # 1. Embedding
        a_emb = self.embedding(premise)  # (batch_size, seq_len, emb_dim)
        b_emb = self.embedding(hypothesis)

        # 2. BiLSTM Encoding (处理变长序列)
        a_encoded = self._masked_lstm(self.encoder, a_emb, premise_mask)  # (batch_size, seq_len, 2*hidden_size)
        b_encoded = self._masked_lstm(self.encoder, b_emb, hypothesis_mask)

        # 3. Attention (应用mask)
        attn = torch.matmul(a_encoded, b_encoded.transpose(1, 2))  # (batch_size, a_len, b_len)

        # 对hypothesis的padding部分加负无穷，softmax后权重为0
        attn = attn.masked_fill(hypothesis_mask.unsqueeze(1) == 0, -1e9)
        a_tilde = torch.matmul(F.softmax(attn, dim=2), b_encoded)

        # 对premise的padding部分加负无穷
        attn = attn.masked_fill(premise_mask.unsqueeze(2) == 0, -1e9)
        b_tilde = torch.matmul(F.softmax(attn, dim=1).transpose(1, 2), a_encoded)

        # 4. Enhancement (拼接特征)
        m_a = torch.cat([a_encoded, a_tilde, a_encoded - a_tilde, a_encoded * a_tilde], dim=2)
        m_b = torch.cat([b_encoded, b_tilde, b_encoded - b_tilde, b_encoded * b_tilde], dim=2)

        # 5. Inference Composition (再次用mask)
        v_a = self._masked_lstm(self.inference_encoder, m_a, premise_mask)
        v_b = self._masked_lstm(self.inference_encoder, m_b, hypothesis_mask)

        # 6. Pooling (只对非padding部分操作)
        v_a = self._masked_pooling(v_a, premise_mask)
        v_b = self._masked_pooling(v_b, hypothesis_mask)
        v = torch.cat([v_a, v_b], dim=1)

        # 7. Classifier
        logits = self.act(self.fc(v))
        return logits

    def _masked_lstm(self, lstm, emb, mask):
        # 处理变长序列：pack -> LSTM -> unpack
        lengths = mask.sum(dim=1).cpu()  # (batch_size,)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=64)
        return output

    def _masked_pooling(self, x, mask):
        # 对非padding部分做max和mean pooling
        mask = mask.unsqueeze(2)  # (batch_size, seq_len, 1)
        x_masked = x * mask  # 将padding置零

        # Max pooling
        max_pool = x_masked.max(dim=1)[0]  # (batch_size, hidden_size)

        # Mean pooling (只对非padding部分求平均)
        sum_pool = x_masked.sum(dim=1)
        cnt = mask.sum(dim=1)  # 非padding的token数
        mean_pool = sum_pool / cnt.clamp(min=1e-9)  # 避免除以零

        return torch.cat([max_pool, mean_pool], dim=1)


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