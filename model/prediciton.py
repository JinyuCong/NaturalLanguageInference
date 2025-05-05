import torch
import torch.nn as nn
from models import ESIMModel, DecomposableAttentionModel
import json
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])


def predict(model: nn.Module, premise: str, hypothesis: str, word_2_index: dict, seq_len: int):

    tokenized_premise = list(nlp(premise.lower()))
    tokenized_hypothesis = list(nlp(hypothesis.lower()))

    premise_index = [word_2_index.get(word_premise.text, 1) for word_premise in tokenized_premise[:seq_len]]
    premise_index += [0 for _ in range(seq_len - len(premise_index))]

    hypothesis_index = [word_2_index.get(word_hypothesis.text, 1) for word_hypothesis in tokenized_hypothesis[:seq_len]]
    hypothesis_index += [0 for _ in range(seq_len - len(hypothesis_index))]

    premise_index = torch.Tensor(premise_index).to(torch.int).unsqueeze(0)
    hypothesis_index = torch.Tensor(hypothesis_index).to(torch.int).unsqueeze(0)

    prediction = model(premise_index, hypothesis_index)
    return prediction


def main():
    with open("word_to_index.json", 'r', encoding='utf-8') as f:
        word_2_index = json.load(f)

    vocab_size = len(word_2_index)
    embedding_dim = 128
    hidden_size = 128
    sequence_length = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ESIMModel(vocab_size, embedding_dim, hidden_size)
    model.load_state_dict(torch.load("./weights/ESIMModel_weights_83.9.pth", map_location=device))
    prediction = predict(model,
                         "Storytelling is necessary for a more sustainable food system because it creates a "
                         "personal relationship and makes the product more appealing to consumers. It adds an element "
                         "of excitement and helps communicate the unique qualities of the food, making it more relatable"
                         " and enticing compared to industrial farming narratives.",
                         "Because storytelling can make the products more attractive to consumers.",
                         word_2_index,
                         sequence_length)

    #prediction = torch.softmax(prediction, dim=1)
    print(prediction)


if __name__ == '__main__':
    main()
