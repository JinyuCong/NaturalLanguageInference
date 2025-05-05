import os.path
import jsonlines
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import ESIMModel
from dataset import NLIDataset
import argparse
import spacy
import json
from datasets import load_dataset

mnli_dataset = load_dataset("glue", "mnli")

train_data = mnli_dataset["train"]
validation_data = mnli_dataset["validation_matched"]
test_data = mnli_dataset["test_matched"]

if __name__ == '__main__':
    print(mnli_dataset)