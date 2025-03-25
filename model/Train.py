import jsonlines
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ESIMModel import ESIMModel
from Dataset import TextDataset


class EarlyStopping:
    def __init__(self, model, verbose, patience=5, delta=0):
        self.model = model
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.patience_counter = 0
        self.best_val_acc = -torch.inf
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_acc):
        # in the case of validation accuracy do not improve
        if val_acc < self.best_val_acc - self.delta:
            self.patience_counter += 1
            if self.verbose:
                print(f"Validation accuracy did not improve. Patience: {self.patience_counter}/{self.patience}")
            if self.patience_counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_acc = val_acc
            self.patience_counter = 0
            self.best_model_weights = self.model.state_dict().copy()


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


def train_with_early_stopping(
        model,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion,
        optimizer,
        device: torch.device,
        epochs: int,
):
    model.to(device)
    early_stopping = EarlyStopping(model, verbose=True)

    for epoch in range(epochs):

        model.train()
        train_loss, train_total = 0, 0
        for pre, hypo, label in train_loader:
            pre, hypo, label = pre.to(device), hypo.to(device), label.to(device)
            prediction = model(pre, hypo)
            loss = criterion(prediction, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() * pre.size(0)
            train_total += pre.size(0)

        model.eval()
        test_correct, test_total = 0, 0
        for test_pre, test_hypo, test_label in test_loader:
            test_pre, test_hypo, test_label = test_pre.to(device), test_hypo.to(device), test_label.to(device)
            prediction = model(test_pre, test_hypo)
            prediction = prediction.argmax(dim=1)
            test_correct += int(torch.sum(prediction == test_label).item())
            test_total += test_label.size(0)

        epoch_loss = train_loss / train_total
        epoch_acc = test_correct / test_total

        print(f'Epoch [{epoch + 1}/{epochs}]: Train loss: {epoch_loss:.4f} | '
              f'Test accuracy: {epoch_acc * 100:.4f}%')

        early_stopping(epoch_acc)
        if early_stopping.early_stop:
            print('Early stopping triggerd, stop training.')
            break

    # torch.save(early_stopping.best_model_weights, "./best_weights.pth")


def main():
    raw_data = build_raw_data("./snli_1.0_dev.jsonl")

    word_2_index, vocab_size = build_word_2_index(raw_data)

    train_data = raw_data[:int(len(raw_data)*0.8)]
    test_data = raw_data[int(len(raw_data)*0.8):]

    batch_size = 32
    seq_len = 32
    emb_dim = 128
    hidden_size = 128
    epochs = 30
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TextDataset(train_data, word_2_index, seq_len)
    test_dataset = TextDataset(test_data, word_2_index, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = ESIMModel(vocab_size, emb_dim, hidden_size)

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    train_with_early_stopping(model, train_loader, test_loader, criterion, opt, device, epochs)


if __name__ == "__main__":
    main()
