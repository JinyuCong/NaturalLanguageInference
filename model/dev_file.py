import jsonlines
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataset import TextDataset
from models import LSTMAttentionEntailment
import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def build_raw_data(jsonl_path: str, batch_size: int = 1000):
    """
    Read the jsonl file and preprocess the premise and hypothesis.
    raw_data is like: [(this is the premise, this is the hypothesis, label)], every premise and hypothesis is tokenized by nlp
    param:
        jsonl_path: path to jsonl file
    return:
        raw_data: list of tuple, each tuple is (premise, hypothesis, label)
    """
    raw_data = []
    premises, hypotheses, labels = [], [], []

    with open(jsonl_path, "r", encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            if line["gold_label"] == "-":
                continue
            premises.append(line["sentence1"].lower())
            hypotheses.append(line["sentence2"].lower())
            labels.append(line["gold_label"])

            # 批量处理
            if len(premises) >= batch_size:
                tokenized_prems = list(nlp.pipe(premises))
                tokenized_hypos = list(nlp.pipe(hypotheses))
                raw_data.extend(zip(tokenized_prems, tokenized_hypos, labels))
                premises, hypotheses, labels = [], [], []

    # 处理剩余数据
    if premises:
        tokenized_prems = list(nlp.pipe(premises))
        tokenized_hypos = list(nlp.pipe(hypotheses))
        raw_data.extend(zip(tokenized_prems, tokenized_hypos, labels))

    return raw_data


def build_word_2_index(raw_data: list[tuple]) -> dict[str, int]:
    """
    Build the word to index dictionary
    param:
        raw_data: raw_data is like: [("this is the premise", "this is the hypothesis", "label")]
    return:
        word_2_index: dictionary {word: index},
        vocab_size: number of vocabs (length of word_2_index)
    """
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    for nli_pair in raw_data:
        tokenized_premise = nli_pair[0]
        tokenized_hypothesis = nli_pair[1]

        for word in tokenized_premise:
            word_2_index[word.text] = word_2_index.get(word.text, len(word_2_index))
        for word in tokenized_hypothesis:
            word_2_index[word.text] = word_2_index.get(word.text, len(word_2_index))

    return word_2_index


raw_data = build_raw_data("../data/snli_1.0/snli_1.0_dev.jsonl")
word_2_index = build_word_2_index(raw_data)
vocab_size = len(word_2_index)

train_data = raw_data[:int(len(raw_data)*0.8)]
test_data = raw_data[int(len(raw_data)*0.8):]

batch_size = 32
seq_len = 64
emb_dim = 300
proj_dim = 200
hidden_size = 256
epochs = 20
learning_rate = 0.001

train_dataset = TextDataset(train_data, word_2_index, seq_len)
test_dataset = TextDataset(test_data, word_2_index, seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model = LSTMAttentionEntailment(vocab_size, emb_dim, hidden_size, 3)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
# first_data = train_dataset[0]
# pre, hypo, label = first_data[0].unsqueeze(0), first_data[1].unsqueeze(0), first_data[2]
# print(pre.shape, hypo.shape) (1, 64) (1, 64)

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for pre, hypo, label in train_loader:
        pre, hypo, label = pre.to(device), hypo.to(device), label.to(device)

        # Forward pass
        outputs = model(pre, hypo)
        loss = criterion(outputs, label)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        train_loss += loss.item() * pre.size(0)
        train_correct += (outputs.argmax(dim=1) == label).sum().item()
        train_total += label.size(0)

    # Evaluation phase
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    with torch.no_grad():
        for test_pre, test_hypo, test_label in test_loader:
            test_pre, test_hypo, test_label = test_pre.to(device), test_hypo.to(device), test_label.to(device)

            # Forward pass
            outputs = model(test_pre, test_hypo)
            test_loss += criterion(outputs, test_label).item() * test_pre.size(0)
            test_correct += (outputs.argmax(dim=1) == test_label).sum().item()
            test_total += test_label.size(0)

    # Calculate epoch metrics
    train_loss = train_loss / train_total
    train_acc = train_correct / train_total
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total

    # Print metrics
    print(
        f'Epoch [{epoch + 1}/{epochs}]: '
        f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | '
        f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%'
    )
