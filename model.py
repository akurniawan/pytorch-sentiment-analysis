import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.autograd import Variable


def _get_rnn_last_output(seq_len, output):
    """Get the last output from RNN model. In classification task,
    it is quite common to only get the last output of RNN and
    put it into the output layer (softmax). Sometimes using output[-1]
    to get the last output may be enough. However, since we do padding
    in every batch, with output[-1] we may end up calculating the wrong
    output. This function responsible to get the exact last output based
    on the sentence length on every batch.
    """
    last_output = torch.index_select(output, 0, seq_len - 1)
    tmp_indices = torch.LongTensor(range(seq_len.size(0)))
    tmp_indices = tmp_indices.view(-1, 1, 1).expand(
        last_output.size(0), 1, last_output.size(2))
    last_output = torch.gather(last_output, 1, tmp_indices)

    return Variable(last_output.squeeze(1))


class ClassifierModel(nn.Module):
    def __init__(self, hidden_size, num_layers, embedding_dim, dropout,
                 vocab_size, label_size):
        super(ClassifierModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False)
        self.dense = nn.Linear(
            in_features=hidden_size, out_features=label_size)
        self.drop = nn.Dropout()

    def forward(self, entity_ids, seq_len):
        embedding = self.embedding(entity_ids)
        out, _ = self.lstm(
            embedding.view(
                embedding.size(1), embedding.size(0), embedding.size(2)))
        # Since we are doing classification, we only need the last
        # output from RNN
        last_output = _get_rnn_last_output(seq_len, out.data)
        logits = self.dense(last_output)
        return logits
