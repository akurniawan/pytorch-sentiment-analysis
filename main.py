import argparse
import multiprocessing

import dataset
import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from dataset import LazyTextDataset, DataLoader
from model import ClassifierModel
from metrics import *

PARSER = argparse.ArgumentParser(
    description="Twitter Sentiment Analysis with char-rnn")
PARSER.add_argument(
    "--epochs", type=int, default=10000, help="Number of epochs")
PARSER.add_argument(
    "--log_every",
    type=int,
    default=100,
    help="""For every n_steps will print out the following logs
    Steps: [<steps>/<epoch>] 'loss: <loss>, accuracy: <acc>""")
PARSER.add_argument(
    "--input_config",
    type=str,
    default="config/input.yml",
    help="Location of the training data")
PARSER.add_argument(
    "--model_config",
    type=str,
    default="config/rnn.yml",
    help="Location of model config")
ARGS = PARSER.parse_args()

if __name__ == "__main__":
    # Load necessary configs
    model_config = load_yaml(ARGS.model_config)
    input_config = load_yaml(ARGS.input_config)

    # Load Dataset
    train_dataset = LazyTextDataset(input_config["path"])
    train_data = DataLoader(
        dataset=train_dataset,
        batch_size=input_config["batch_size"],
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        collate_fn=dataset.collate_fn)

    # Build model graph
    classifier_model = ClassifierModel(
        hidden_size=model_config["nhidden"],
        num_layers=model_config["nlayers"],
        embedding_dim=model_config["nembedding"],
        dropout=model_config["dropout"],
        vocab_size=train_dataset.vocab_size,
        label_size=train_dataset.category_size)
    classifier_model = maybe_use_cuda(classifier_model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters())
    steps = 0

    for epoch in range(ARGS.epochs):
        for data_bucket, category_bucket, seq_len_bucket in train_data:
            sum_loss_bucket = 0.0
            avg_acc_bucket = 0.0
            total_bucket = 0
            for keys in data_bucket.keys():
                # TODO: try to update collate_fn to use yield
                entity_ids = maybe_use_cuda(data_bucket[keys])
                category_ids = maybe_use_cuda(category_bucket[keys])
                seq_len = seq_len_bucket[keys]

                # Perform model training
                logits = classifier_model(entity_ids, seq_len)
                loss = loss_fn(logits, category_ids)
                sum_loss_bucket += loss

                avg_acc_bucket += accuracy(logits, category_ids)

                # Update parameters by performing backpropagation
                loss.backward()
                # To prevent exploding/vanishing gradient problem which commonly
                # occurs in RNN model
                nn.utils.clip_grad_norm(classifier_model.parameters(),
                                        model_config["clip"])
                optimizer.step()
                total_bucket += 1

            if steps % ARGS.log_every == 0:
                print("Steps: [%d/%d] loss: %.5f, accuracy: %.5f" %
                      (steps, epoch + 1, sum_loss_bucket.data.numpy()[0],
                       avg_acc_bucket / total_bucket))

            steps += 1
