from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer
import torch.nn as nn
from datasets import load_dataset


class BertForEvidenceSpan(nn.Module):
    def __init__(self, config):
        super(BertForEvidenceSpan, self).__init__()
        self.bert = BertModel(config)
        self.qa_start = nn.Linear(config.hidden_size, 1)
        self.qa_end = nn.Linear(config.hidden_size, 1)
        self.evidence_start = nn.Linear(config.hidden_size, 1)
        self.evidence_end = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        print(outputs)


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = BertForEvidenceSpan(config)

    squad = load_dataset('squad_v2')
    test = squad['train'][100]
    print(test)


