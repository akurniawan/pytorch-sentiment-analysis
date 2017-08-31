import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset


class ClassifierModel(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 vocab_size,
                 embedding_dim,
                 label_size,
                 pretrained_embedding=None):
        super(ClassifierModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.5,
            bidirectional=False)
        self.dense = nn.Linear(
            in_features=hidden_size, out_features=label_size)

    def forward(self, word_ids):
        embedding = self.embedding(word_ids)
        print(embedding.size())
        input_size = embedding.size()
        out, _ = self.lstm(embedding.view(input_size[1], input_size[0], input_size[2]))
        logits = self.dense(out[-1, :, :])
        logits = F.relu(logits)
        return logits
