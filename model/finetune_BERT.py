import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
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
        scheduler,
        device: torch.device,
        epochs: int,
) -> None:
    """
    Train the model with early stopping, tracking train/test accuracy and loss.
    :param model: model to train
    :param train_loader: DataLoader for training set
    :param test_loader: DataLoader for test set
    :param optimizer: optimizer
    :param device: cuda or cpu
    :param epochs: number of epochs
    :return: None
    """
    model.to(device)
    early_stopping = EarlyStopping(model, verbose=True)

    step_losses = []
    epoch_accs = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')
        for batch in progress_bar:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Metrics
            total_loss += loss.item()
            step_losses.append(loss.item())
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # 验证
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1)
                total_correct += (predictions == inputs['labels']).sum().item()
                total_samples += inputs['labels'].size(0)

        accuracy = total_correct / total_samples
        epoch_accs.append(accuracy)
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Early stopping checking by test accuracy
        early_stopping(accuracy)
        if early_stopping.early_stop:
            print('Early stopping triggered.')
            break

    plot_stat(step_losses, f"Fine-tuned BERT per step losses on snli", "step", "loss")
    plot_stat(epoch_accs, f"Fine-tuned BERT epoch accuracies on snli", "epoch", "accuracy")

    # Save best model weights
    torch.save(early_stopping.best_model_weights, f"./weights/{model._get_name()}_weights_{early_stopping.best_val_acc * 100:.2f}.pth")


def plot_stat(stat: list, title: str, x_label: str, y_label: str) -> None:
    plt.plot(stat)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f"./plots/{title}.png")
    plt.show()


def main(
        batch_size: int,
        max_len: int,
        epochs: int,
        learning_rate: float,
        snli_or_mnli: str
):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def preprocess(example):
        encoding = tokenizer(
            example['premise'],
            example['hypothesis'],
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['labels'] = torch.tensor(example['label'])
        return encoding

    snli_dataset = load_dataset("snli")
    mnli_dataset = load_dataset("multi_nli")

    if snli_or_mnli == "snli":
        train_data = snli_dataset['train'].filter(lambda x: x['label'] != -1).map(preprocess)
        valid_data = snli_dataset['test'].filter(lambda x: x['label'] != -1).map(preprocess)
    elif snli_or_mnli == "mnli":
        train_data = mnli_dataset['train'].filter(lambda x: x['label'] != -1).map(preprocess)
        valid_data = mnli_dataset['validation_matched'].filter(lambda x: x['label'] != -1).map(preprocess)

    train_data = train_data.with_format("torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    valid_data = valid_data.with_format("torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

    # bert模型的输入需要是这个格式的
    # train_data:
    # {
    #   'input_ids': tensor([101 ... 102 ... 102 0 0 ...]),
    #   'token_type_ids': tensor([0 0 ... 1 1 1 ... 0 0 ...])
    #   'attention_mask': tensor([1 1 ... 0 0 ...])
    #   'labels': tensor([0|1|2])
    # }

    # dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    # define BERT model without pretrained
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3
    ).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    train_with_early_stopping(model, train_loader, valid_loader, optimizer, scheduler, device, epochs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train ESIM Natural Language Inference models.")

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["snli", "mnli"],
        help="Use snli dataset or mnli dataset to train",
        default="snli"
    )

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
        default=128
    )
    parser.add_argument(
        "-E",
        "--epochs",
        type=int,
        help="Number of epochs",
        default=3
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        help="Learning rate",
        default=2e-5
    )

    args = parser.parse_args()

    dataset = args.dataset
    batch_size = args.batch_size
    max_length = args.max_length
    epochs = args.epochs
    learning_rate = args.learning_rate

    main(
        batch_size=batch_size,
        max_len=max_length,
        epochs=epochs,
        learning_rate=learning_rate,
        snli_or_mnli=dataset
    )
