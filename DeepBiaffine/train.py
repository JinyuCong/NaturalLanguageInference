import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import ConlluDataset
from DeepBiaffine import DeepBiaffineDependencyParser

# define max seq len and batch_size
max_len = 64
batch_size = 32

# define the train and test dataset
train_dataset = ConlluDataset('SUD_English-EWT/en_ewt-sud-train.conllu', max_len=max_len)
test_dataset = ConlluDataset('SUD_English-EWT/en_ewt-sud-test.conllu', max_len=max_len)

# train dataloader and test data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# hyper parameters
epochs = 50
vocab_size = len(train_dataset.word_2_index)
pos_size = len(test_dataset.pos_2_index)
embedding_dim = 256
lstm_hidden_size = 256
mlp_dim = 256
arc_out = 1
label_out = len(train_dataset.deprel_2_index)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define de model optimizer and loss function
model = DeepBiaffineDependencyParser(vocab_size, pos_size, embedding_dim, lstm_hidden_size, mlp_dim, arc_out, label_out).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# main loop
for epoch in range(epochs):
    # train_phase
    model.train()
    train_loss, train_total = 0, 0
    for word_ids, pos_ids, arc_labels, rel_labels in train_loader:
        word_ids, pos_ids, arc_labels, rel_labels = word_ids.to(device), pos_ids.to(device), arc_labels.to(device), rel_labels.to(device)

        # forward pass
        arc_scores, rel_scores = model(word_ids, pos_ids)
        arc_loss = criterion(arc_scores, arc_labels)

        valid_positions = (rel_labels != -1)
        valid_rel_scores = rel_scores[valid_positions]
        valid_rel_labels = rel_labels[valid_positions]
        rel_loss = criterion(valid_rel_scores, valid_rel_labels)

        # calculate both of the arc and the relation loss
        both_loss = arc_loss + rel_loss

        # backward
        optimizer.zero_grad()
        both_loss.backward()
        optimizer.step()

        train_loss += rel_loss.item() * word_ids.size(0)
        train_total += word_ids.size(0)

    # evaluation phase
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for test_word_ids, test_pos_ids, test_arc_labels, test_rel_labels in test_loader:
            test_word_ids, test_pos_ids, test_arc_labels, test_rel_labels = test_word_ids.to(device), test_pos_ids.to(device), test_arc_labels.to(device), test_rel_labels.to(device)

            test_arc_scores, test_rel_scores = model(test_word_ids, test_pos_ids)
            test_correct += (test_rel_scores.argmax(dim=-1) == test_rel_labels).sum().item()  # 这里需要修改
            test_total += test_arc_scores.size(0)

    train_loss = train_loss / train_total
    test_acc = test_correct / test_total
    # Print metrics
    print(
        f'Epoch [{epoch + 1}/{epochs}]: '
        f'Train Loss: {train_loss:.4f} | Test Acc: {test_acc * 100:.2f}%'
    )


'''
        arc_scores = arc_scores.reshape(arc_scores.size(0), -1)
        arc_label = arc_label.reshape(arc_label.size(0), -1)  # [B, S*S]
        print(arc_scores.size(), arc_label.size())
        #arc_loss = criterion(arc_scores, arc_label)

        label_scores = label_scores.reshape(-1, label_scores.size(-1))  # [B*S*S, num_labels]
        rel_label = rel_label.reshape(-1)  # [B*S*S]
        print(label_scores.size(), rel_label.size())
        #label_loss = criterion(label_scores, rel_label)
'''

