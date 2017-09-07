import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils import maybe_use_cuda

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
    tmp_indices = maybe_use_cuda(torch.LongTensor(range(seq_len.size(0))))
    tmp_indices = tmp_indices.view(-1, 1, 1).expand(
        last_output.size(0), 1, last_output.size(2))
    last_output = torch.gather(last_output, 1, tmp_indices)

    return Variable(last_output.squeeze(1))


class RNNClassifier(nn.Module):
    def __init__(self, config, vocab_size, label_size):
        super(RNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, config["nembedding"], padding_idx=0)
        self.gru = nn.GRU(
            input_size=config["nembedding"],
            hidden_size=config["nhidden"],
            num_layers=config["nlayers"],
            dropout=config["dropout"],
            bidirectional=False)
        self.dense = nn.Linear(
            in_features=config["nhidden"], out_features=label_size)

    def forward(self, entity_ids, seq_len):
        embedding = self.embedding(entity_ids)
        out, _ = self.gru(
            embedding.view(
                embedding.size(1), embedding.size(0), embedding.size(2)))
        # Since we are doing classification, we only need the last
        # output from RNN
        last_output = _get_rnn_last_output(seq_len, out.data)
        logits = self.dense(last_output)
        return logits


class CNNRNNClassifier(nn.Module):
    def __init__(self, config, vocab_size, label_size):
        super(CNNRNNClassifier, self).__init__()
        cnn_config = config["cnn"]
        rnn_config = config["rnn"]

        self.embedding = nn.Embedding(vocab_size, config["nembedding"], padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, Nk, Ks)
            for Ks, Nk in zip(cnn_config["kernel_sizes"], cnn_config[
                "nkernels"])
        ])
        self.lstm = nn.LSTM(
            input_size=cnn_config["nkernels"][-1],
            hidden_size=rnn_config["nhidden"],
            num_layers=rnn_config["nlayers"],
            dropout=rnn_config["dropout"],
            bidirectional=False)
        self.dense = nn.Linear(
            in_features=rnn_config["nhidden"], out_features=label_size)

    def forward(self, entity_ids, seq_len):
        x = self.embedding(entity_ids)

        # Since we are using conv2d, we need to add extra outer dimension
        for i, conv in enumerate(self.convs):
            x = x.unsqueeze(1)
            x = F.relu(conv(x)).squeeze(3)
            x = x.view(x.size(0), x.size(2), x.size(1))

        out, _ = self.lstm(x.view(x.size(1), x.size(0), x.size(2)))
        last_output = out[-1, :, :]
        logits = self.dense(last_output)

        return logits
