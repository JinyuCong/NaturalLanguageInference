import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, raw_data, word_2_index, seq_len):
        self.raw_data = raw_data
        self.seq_len = seq_len
        self.word_2_index = word_2_index
        self.index_2_word = {index: word for word, index in self.word_2_index.items()}
        self.label_mapping = {"neutral": 0, "entailment": 1, "contradiction": 2}
        self.data = self._build_corpus()

    def _build_corpus(self):
        data = []
        for premise, hypothesis, label in self.raw_data:
            premise, hypothesis, label = premise.lower(), hypothesis.lower(), label.lower()

            premise_index = [self.word_2_index.get(word_premise, 1) for word_premise in premise.split()[:self.seq_len]]
            premise_index += [0 for _ in range(self.seq_len - len(premise_index))]

            hypothesis_index = [self.word_2_index.get(word_hypothesis, 1) for word_hypothesis in hypothesis.split()[:self.seq_len]]
            hypothesis_index += [0 for _ in range(self.seq_len - len(hypothesis_index))]

            label_index = self.label_mapping[label]
            data.append((premise_index, hypothesis_index, label_index))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise_index = self.data[idx][0]
        hypothesis_index = self.data[idx][1]
        label_index = self.data[idx][2]
        return torch.Tensor(premise_index).to(torch.int), torch.Tensor(hypothesis_index).to(torch.int), label_index