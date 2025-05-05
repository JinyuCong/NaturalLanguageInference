import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


class NLIDataset(Dataset):
    """
    raw_data is like: [(this is the premise, this is the hypothesis, label)]
    """
    def __init__(self, raw_dataset, tokenizer, max_len):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        self.max_len = max_len
        self.data = self._build_corpus()

    def _build_corpus(self):
        premises = [nli_pair['premise'] for nli_pair in self.raw_dataset]
        hypotheses = [nli_pair['hypothesis'] for nli_pair in self.raw_dataset]
        labels = [nli_pair['label'] for nli_pair in self.raw_dataset]

        # 批量编码 premise 和 hypothesis
        premise_encodings = self.tokenizer(
            premises,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )
        hypothesis_encodings = self.tokenizer(
            hypotheses,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        )

        data = []
        for pre_ids, hypo_ids, label in zip(premise_encodings['input_ids'], hypothesis_encodings['input_ids'], labels):
            if label == -1:
                continue

            data.append((
                pre_ids,
                hypo_ids,
                torch.tensor(label)
            ))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise_ids = self.data[idx][0]
        hypothesis_ids = self.data[idx][1]
        label = self.data[idx][2]
        return premise_ids, hypothesis_ids, label


if __name__ == '__main__':
    snli_dataset = load_dataset("snli")

    train_dataset = snli_dataset["train"]
    test_dataset = snli_dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    vocab_size = tokenizer.vocab_size
    max_len = 64

    dataset = NLIDataset(test_dataset, tokenizer, max_len)

    for premise, hypothesis, label in dataset:
        print(label)
