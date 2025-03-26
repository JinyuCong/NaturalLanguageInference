import jsonlines
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Dataset import TextDataset
from ESIMModel import DecomposableAttentionModel, ESIMModel


def build_raw_data(jsonl_path: str):
    """
    raw_data is like: [("this is the premise", "this is the hypothesis", "label")]
    :param jsonl_path: path to jsonl file
    :return raw_data: list of tuple, each tuple is (premise, hypothesis, label)
    """
    raw_data = []
    with open(jsonl_path, "r", encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            premise = line["sentence1"]
            hypothesis = line["sentence2"]
            label = line["gold_label"]
            if label == "-":
                continue
            raw_data.append((premise, hypothesis, label))
    return raw_data


def build_word_2_index(raw_data: list[tuple]) -> tuple[dict[str, int], int]:
    """

    :param raw_data: raw_data is like: [("this is the premise", "this is the hypothesis", "label")]
    :return: word_2_index: dictionary {word: index}, vocab_size: number of vocabs
    """
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    for nli_pair in raw_data:
        premise, hypothesis = nli_pair[0].lower().split(), nli_pair[1].lower().split()
        for word in premise:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
        for word in hypothesis:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))

    vocab_size = len(word_2_index)
    return word_2_index, vocab_size


raw_data = build_raw_data("../snli_1.0/snli_1.0_dev.jsonl")
word_2_index, vocab_size = build_word_2_index(raw_data)

batch_size = 32
seq_len = 64
emb_dim = 128
hidden_size = 128
epochs = 10
learning_rate = 0.001

dataset = TextDataset(raw_data, word_2_index, seq_len)
model = DecomposableAttentionModel(vocab_size, emb_dim, hidden_size)
dataloader = DataLoader(dataset, batch_size=batch_size)

test_data = dataset[0]
pre, hypo, label = test_data[0].unsqueeze(0), test_data[1].unsqueeze(0), test_data[2]

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(epochs):

    model.train()
    train_loss, train_total = 0, 0
    for pre, hypo, label in dataloader:
        prediction = model(pre, hypo)
        loss = criterion(prediction, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item() * pre.size(0)
        train_total += pre.size(0)

    epoch_loss = train_loss / train_total

    print(f'Epoch [{epoch + 1}/{epochs}]: Train loss: {epoch_loss:.4f}')