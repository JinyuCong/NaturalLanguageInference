import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import ESIMModel
from dataset import NLIDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse


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


def train_with_early_stopping(
        model,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer,
        device: torch.device,
        epochs: int,
        criterion,  # 新增：损失函数参数
) -> None:
    """
    Train the model with early stopping, tracking train/test accuracy and loss.
    :param model: model to train
    :param train_loader: DataLoader for training set
    :param test_loader: DataLoader for test set
    :param optimizer: optimizer
    :param device: cuda or cpu
    :param epochs: number of epochs
    :param criterion: loss function (e.g., nn.CrossEntropyLoss())
    :return: None
    """
    model.to(device)
    early_stopping = EarlyStopping(model, verbose=True)

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

        # Early stopping checking by test accuracy
        early_stopping(test_acc)
        if early_stopping.early_stop:
            print('Early stopping triggered.')
            break

    # Save best model weights
    torch.save(early_stopping.best_model_weights, f"./weights/{model._get_name()}_weights_{early_stopping.best_val_acc * 100:.2f}.pth")


def main(
        batch_size: int,
        max_length: int,
        embedding_dim: int,
        hidden_size: int,
        epochs: int,
        learning_rate: float,
):

    snli_dataset = load_dataset("snli")

    text_train_dataset = snli_dataset["train"]
    text_test_dataset = snli_dataset["test"]

    # 各项参数
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = tokenizer.vocab_size

    train_dataset = NLIDataset(text_train_dataset, tokenizer, max_length)
    test_dataset = NLIDataset(text_test_dataset, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()

    # define ESIM model
    model = ESIMModel(vocab_size, embedding_dim, hidden_size)

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_with_early_stopping(model, train_loader, test_loader, opt, device, epochs, criterion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ESIM Natural Language Inference models.")

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Batch size",
        default=32
    )
    parser.add_argument(
        "-s",
        "--max_length",
        type=int,
        help="Sequence length",
        default=64
    )
    parser.add_argument(
        "-e",
        "--embedding_dim",
        type=int,
        help="Embedding dimension",
        default=128
    )
    parser.add_argument(
        "-w",
        "--hidden_size",
        type=int,
        help="Hidden size",
        default=128
    )
    parser.add_argument(
        "-E",
        "--epochs",
        type=int,
        help="Number of epochs",
        default=30
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        help="Learning rate",
        default=1e-3
    )

    args = parser.parse_args()

    batch_size = args.batch_size
    max_length = args.max_length
    embedding_dim = args.embedding_dim
    hidden_size = args.hidden_size
    epochs = args.epochs
    learning_rate = args.learning_rate

    main(
        batch_size=batch_size,
        max_length=max_length,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        epochs=epochs,
        learning_rate=learning_rate
    )
