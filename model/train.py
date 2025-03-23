import jsonlines
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import TextDataset, ESIMModel


data = []
with open("../snli_1.0/snli_1.0_dev.jsonl", "r", encoding='utf-8') as f:
    for line in jsonlines.Reader(f):
        premise = line["sentence1"]
        hypothesis = line["sentence2"]
        label = line["gold_label"]
        if label == "-":
            continue
        data.append((premise, hypothesis, label))


batch_size = 32
seq_len = 64
emb_dim = 128
hidden_size = 128
epochs = 10
learning_rate = 0.001

dataset = TextDataset(data, seq_len)
vocab_size = len(dataset.word_2_index)

dataloader = DataLoader(dataset, batch_size=batch_size)

model = ESIMModel(vocab_size, emb_dim, hidden_size)

opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):

    train_loss, train_total = 0, 0
    for pre, hypo, label in dataloader:
        prediction = model(pre, hypo)
        loss = criterion(prediction, label)

        loss.backward()
        opt.step()
        opt.zero_grad()

        train_loss += loss.item() * pre.size(0)
        train_total += pre.size(0)

    epoch_loss = train_loss / train_total

    print(f'Epoch [{epoch + 1}/{epochs}]: Train loss: {epoch_loss:.4f}')