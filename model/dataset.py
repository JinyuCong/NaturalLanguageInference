import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


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
        for i in range(len(labels)):
            if labels[i] == -1:
                continue

            pre_ids = premise_encodings['input_ids'][i]
            pre_mask = premise_encodings['attention_mask'][i]
            hypo_ids = hypothesis_encodings['input_ids'][i]
            hypo_mask = hypothesis_encodings['attention_mask'][i]
            label = torch.tensor(labels[i])

            data.append((pre_ids, pre_mask, hypo_ids, hypo_mask, label))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pre_ids, pre_mask, hypo_ids, hypo_mask, label = self.data[idx]
        return pre_ids, pre_mask, hypo_ids, hypo_mask, label



