import torch
from torch.utils.data import Dataset, DataLoader
from conllu import parse


class ConlluDataset(Dataset):
    def __init__(self, conllu_file_path, max_len):
        self.max_len = max_len
        self.sentences = self._read_conllu(conllu_file_path)
        self.word_2_index, self.pos_2_index, self.deprel_2_index = self._build_all_to_index()
        self.data = self._build_corpus()

    def _read_conllu(self, conllu_file_path: str):
        with open(conllu_file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        return parse(data)

    def _build_all_to_index(self):
        word_to_index = {'<PAD>': 0, '<UNK>:': 1}
        pos_to_index = {'<PAD>': 0, '<UNK>:': 1}
        deprel_to_index = {'<PAD>': -1}

        for sentence in self.sentences:
            for token in sentence:
                word = token['form']
                pos = token['upos']
                deprel = token['deprel']
                word_to_index[word] = word_to_index.get(word, len(word_to_index))
                pos_to_index[pos] = pos_to_index.get(pos, len(pos_to_index))
                deprel_to_index[deprel] = deprel_to_index.get(deprel, len(deprel_to_index))

        return word_to_index, pos_to_index, deprel_to_index

    def _build_corpus(self):
        data = []
        for sentence in self.sentences:
            # 跳过无效行
            valid_tokens = [t for t in sentence if t['head'] is not None and t['deprel'] is not None][:self.max_len]

            word_ids = [self.word_2_index.get(token['form'], 1) for token in valid_tokens]
            pos_tags = [self.pos_2_index.get(token['upos'], 1) for token in valid_tokens]

            word_ids += [0] * (self.max_len - len(word_ids))
            pos_tags += [0] * (self.max_len - len(pos_tags))

            arc = torch.zeros((self.max_len, self.max_len))
            rel = torch.full((self.max_len, self.max_len), -1)

            for i, token in enumerate(valid_tokens):
                head_idx = token['head'] - 1
                if 0 <= head_idx < self.max_len and i < self.max_len:
                    arc[i][head_idx] = 1
                    rel[i][head_idx] = self.deprel_2_index[token['deprel']]

            data.append((word_ids, pos_tags, arc, rel))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word_index = torch.tensor(self.data[idx][0]).to(torch.int)
        pos_index = torch.tensor(self.data[idx][1]).to(torch.int)
        arc_labels = self.data[idx][2]
        rel_labels = self.data[idx][3]
        return word_index, pos_index, arc_labels, rel_labels
