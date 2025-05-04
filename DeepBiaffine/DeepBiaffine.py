import torch
import torch.nn as nn
import torch.nn.functional as F


class Biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.U = nn.Parameter(torch.Tensor(out_size, in_size + int(bias_x), in_size + int(bias_y)))
        nn.init.xavier_uniform_(self.U)

    def forward(self, x, y):  # x: head, y: dep
        if self.bias_x:
            x = torch.cat([x, x.new_ones(x.size(0), x.size(1), 1)], dim=-1)  # 隐式偏置项
        if self.bias_y:
            y = torch.cat([y, y.new_ones(y.size(0), y.size(1), 1)], dim=-1)  # 隐式偏置项
        # x: [batch, seq_len, dim+1], y: [batch, seq_len, dim+1]
        # Output: [batch, seq_len, seq_len, out_size]
        biaffine = torch.einsum('bxi,oij,byj->boxy', x, self.U, y)
        biaffine = biaffine.permute(0, 2, 3, 1)
        return biaffine.squeeze(-1) if biaffine.shape[-1] == 1 else biaffine


class DeepBiaffineDependencyParser(nn.Module):
    def __init__(self, vocab_size, pos_size, embed_dim, lstm_hidden, mlp_dim, arc_out, label_out):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(pos_size, embed_dim)

        self.lstm = nn.LSTM(input_size=embed_dim*2, hidden_size=lstm_hidden,
                            bidirectional=True, batch_first=True, num_layers=3)

        self.mlp_arc_dep = nn.Linear(lstm_hidden * 2, mlp_dim)
        self.mlp_arc_head = nn.Linear(lstm_hidden * 2, mlp_dim)
        self.mlp_label_dep = nn.Linear(lstm_hidden * 2, mlp_dim)
        self.mlp_label_head = nn.Linear(lstm_hidden * 2, mlp_dim)

        self.arc_biaffine = Biaffine(mlp_dim, arc_out)
        self.label_biaffine = Biaffine(mlp_dim, label_out)

    def forward(self, word_ids, pos_ids):
        x_word = self.word_embed(word_ids)  # [batch_size, seq_len, emb_dim]
        x_pos = self.pos_embed(pos_ids)  # [batch_size, seq_len, emb_dim]
        x = torch.cat([x_word, x_pos], dim=-1)  # [batch_size, seq_len, 2*emb_dim]

        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, 2*lstm_hidden]
        arc_dep = F.relu(self.mlp_arc_dep(lstm_out))  # [batch_size, seq_len, mlp_dim]
        arc_head = F.relu(self.mlp_arc_head(lstm_out))  # [batch_size, seq_len, mlp_dim]
        label_dep = F.relu(self.mlp_label_dep(lstm_out))  # [batch_size, seq_len, mlp_dim]
        label_head = F.relu(self.mlp_label_head(lstm_out))  # [batch_size, seq_len, mlp_dim]

        arc_scores = self.arc_biaffine(arc_dep, arc_head)  # [B, seq_len, seq_len]
        label_scores = self.label_biaffine(label_dep, label_head)  # [B, seq_len, seq_len, num_labels]
        return arc_scores, label_scores
