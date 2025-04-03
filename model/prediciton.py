import torch
from models import ESIMModel, DecomposableAttentionModel


def predict(model, premise: str, hypothesis: str, word_2_index: dict[str, int], seq_len: int):
    premise_index = [word_2_index.get(word_premise, 1) for word_premise in premise.split()[:seq_len]]
    premise_index += [0 for _ in range(seq_len - len(premise_index))]

    hypothesis_index = [word_2_index.get(word_hypothesis, 1) for word_hypothesis in hypothesis.split()[:seq_len]]
    hypothesis_index += [0 for _ in range(seq_len - len(hypothesis_index))]

    premise_index = torch.Tensor(premise_index).to(torch.int).to('cuda').unsqueeze(0)
    hypothesis_index = torch.Tensor(hypothesis_index).to(torch.int).to('cuda').unsqueeze(0)

    prediction = model(premise_index, hypothesis_index)
    return prediction


def main():
    pass


if __name__ == '__main__':
    pass
