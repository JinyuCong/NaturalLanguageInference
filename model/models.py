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


# ------------------------ESIM model LSTM replacement with multihead attention--------------------
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


# ---------------------------------------DRCN model-----------------------------------------
class CharCNN(torch.nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, out_channels, kernel_size):
        super(CharCNN, self).__init__()
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim)
        self.conv = nn.Conv1d(in_channels=char_emb_dim, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x):
        # x: (batch_size, seq_len, word_len)
        batch_size, seq_len, word_len = x.shape
        x = x.view(-1, word_len)  # (batch_size * seq_len, word_len) torch.Size([128, 32])
        x = self.char_emb(x)  # (batch_size * seq_len, word_len, char_emb_dim) torch.Size([128, 32, 16])
        x = torch.transpose(x, 1, 2)  # (batch_size * seq_len, char_emb_dim, word_len) torch.Size([128, 16, 32])
        x = self.conv(x)  # (batch_size * seq_len, out_channels, L_out)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(
            2)  # (batch_size * seq_len, out_channels) torch.Size([128, 32])
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, out_channels) torch.Size([2, 64, 32])
        return x


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.out(x)
        return x  # 注意这里不要 softmax，交给 nn.CrossEntropyLoss 来做


class DRCN(torch.nn.Module):
    def __init__(self, word_vocab_size, word_emb_dim, char_vocab_size, char_emb_dim, char_out_channels,
                 char_kernel_size,
                 hidden_size, num_layers, pretrained_word_emb=None, trainable_emb=True):
        super(DRCN, self).__init__()
        # word embeddings
        self.word_emb_fix = nn.Embedding(word_vocab_size, word_emb_dim)
        self.word_emb_tr = nn.Embedding(word_vocab_size, word_emb_dim)

        if pretrained_word_emb is not None:
            self.word_emb_fix.weight.data.copy_(pretrained_word_emb)
            self.word_emb_fix.weight.requires_grad = False
            self.word_emb_tr.weight.data.copy_(pretrained_word_emb)
            self.word_emb_tr.weight.requires_grad = trainable_emb

        # Char CNN
        self.char_cnn = CharCNN(char_vocab_size, char_emb_dim, char_out_channels, char_kernel_size)

        # Densely connected Recurrent Networks
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn_layers = nn.ModuleList([
            nn.LSTM(input_size=(2 * word_emb_dim + char_out_channels + 1) + l * (2*hidden_size + 2*hidden_size),
                    hidden_size=hidden_size,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True)
            for l in range(num_layers)
        ])

    def forward(self, word_ids_p, word_ids_q, char_ids_p, char_ids_q, match_flag_p, match_flag_q):
        # word_ids: (batch_size, seq_len); char_ids: (batch_size, seq_len, word_len)
        # 处理句子p，也就是premise
        word_emb_fix_p = self.word_emb_fix(word_ids_p)  # (batch_size, seq_len_p, word_emb_dim)
        word_emb_tr_p = self.word_emb_tr(word_ids_p)  # (batch_size, seq_len_p, word_emb_dim)
        char_emb_p = self.char_cnn(char_ids_p)  # (batch_size, seq_len_p, char_out_channels)
        match_flag_p = match_flag_p.unsqueeze(-1).float()  # (batch_size, seq_len_p, 1)

        rep_p = torch.concat([word_emb_tr_p, word_emb_fix_p, char_emb_p, match_flag_p], dim=-1)

        # 处理句子q，也就是hypothesis
        word_emb_fix_q = self.word_emb_fix(word_ids_q)
        word_emb_tr_q = self.word_emb_tr(word_ids_q)
        char_emb_q = self.char_cnn(char_ids_q)
        match_flag_q = match_flag_q.unsqueeze(-1).float()

        rep_q = torch.concat([word_emb_tr_q, word_emb_fix_q, char_emb_q, match_flag_q], dim=-1)

        # ----- Densely Connected BiLSTM + Co-Attention -----
        dense_p = rep_p
        dense_q = rep_q

        for rnn_layer in self.rnn_layers:
            out_p, _ = rnn_layer(dense_p)  # (batch_size, seq_len_p, 2*hidden_size)
            out_q, _ = rnn_layer(dense_q)  # (batch_size, seq_len_q, 2*hidden_size)

            # Co-Attention 和 cosine similarity attention
            attention_weights = F.cosine_similarity(out_p.unsqueeze(2), out_q.unsqueeze(1), dim=-1)
            attn_p2q = F.softmax(attention_weights, dim=-1)  # (batch_size, seq_len_p, seq_len_q)
            attn_q2p = F.softmax(attention_weights.transpose(1, 2), dim=-1)  # (batch_size, seq_len_q, seq_len_p)

            context_p = torch.bmm(attn_p2q, out_q)  # (batch, seq_len_p, 2 * hidden_size)
            context_q = torch.bmm(attn_q2p, out_p)  # (batch, seq_len_q, 2 * hidden_size)

            # Dense connection: concatenate current layer output and co-attentive context
            dense_p = torch.cat([dense_p, out_p, context_p], dim=-1)
            dense_q = torch.cat([dense_q, out_q, context_q], dim=-1)

        final_p = torch.max(dense_p, dim=1)[0]
        final_q = torch.max(dense_q, dim=1)[0]

        v = torch.concat([final_p, final_q, final_p + final_q, final_p - final_q, torch.abs(final_p - final_q)], dim=-1)

        # classifier
        cls = MLPClassifier(input_dim=v.size(1), hidden_dim=1000, output_dim=3)
        logits = cls(v)

        return logits


if __name__ == '__main__':
    batch_size = 16
    seq_len_p = 30
    seq_len_q = 28
    word_len = 10
    word_vocab_size = 5000
    char_vocab_size = 100
    word_emb_dim = 300
    char_emb_dim = 16
    char_out_channels = 32
    char_kernel_size = 5
    hidden_size = 100
    num_layers = 5

    model = DRCN(word_vocab_size, word_emb_dim, char_vocab_size, char_emb_dim, char_out_channels,
                 char_kernel_size, hidden_size, num_layers)

    word_ids_p = torch.randint(0, word_vocab_size, (batch_size, seq_len_p))
    word_ids_q = torch.randint(0, word_vocab_size, (batch_size, seq_len_q))
    char_ids_p = torch.randint(0, char_vocab_size, (batch_size, seq_len_p, word_len))
    char_ids_q = torch.randint(0, char_vocab_size, (batch_size, seq_len_q, word_len))
    match_flag_p = torch.randint(0, 2, (batch_size, seq_len_p))
    match_flag_q = torch.randint(0, 2, (batch_size, seq_len_q))

    model(word_ids_p, word_ids_q, char_ids_p, char_ids_q, match_flag_p, match_flag_q)
