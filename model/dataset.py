import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


class NLIDataset(Dataset):
    """
    raw_data is like: {'premise': 'this is the premise', 'hypothesis': 'this is the hypothesis', 'label': 0|1|2}
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


class DRCNDataset(Dataset):
    """
    raw_data is like: [{'premise': 'this is the premise', 'hypothesis': 'this is the hypothesis', 'label': 0|1|2}]
    """
    def __init__(self, raw_dataset, tokenizer, seq_len):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        self.seq_len = seq_len
        self.char_to_index = {}
        self.data = self._build_corpus()

    def _build_corpus(self):
        premises = [nli_pair['premise'] for nli_pair in self.raw_dataset]
        hypotheses = [nli_pair['hypothesis'] for nli_pair in self.raw_dataset]
        labels = [nli_pair['label'] for nli_pair in self.raw_dataset]

        # 批量编码 premise 和 hypothesis
        premise_encodings = self.tokenizer(
            premises,
            padding='max_length',
            max_length=self.seq_len,
            truncation=True,
            return_tensors="pt",
        )
        hypothesis_encodings = self.tokenizer(
            hypotheses,
            padding='max_length',
            max_length=self.seq_len,
            truncation=True,
            return_tensors="pt",
        )
        data = []
        for i in range(len(labels)):
            if labels[i] == -1:
                continue

            word_ids_pre = premise_encodings['input_ids'][i]
            pre_text = tokenizer.convert_ids_to_tokens(word_ids_pre)
            word_ids_hypo = hypothesis_encodings['input_ids'][i]
            hypo_text = tokenizer.convert_ids_to_tokens(word_ids_hypo)
            label = torch.tensor(labels[i])

            for word_pre in pre_text:
                for char_pre in word_pre:
                    if char_pre not in self.char_to_index:
                        self.char_to_index[char_pre] = len(self.char_to_index)

        print(self.char_to_index)



if __name__ == "__main__":
    snli_dataset = load_dataset("snli")
    text_test_dataset = snli_dataset['test']
    test_data = [text_test_dataset[0]]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word_vocab_size = tokenizer.vocab_size
    seq_len = 64

    drcn_dataset = DRCNDataset(test_data, tokenizer, seq_len)
