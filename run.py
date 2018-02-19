import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

from ignite.engine import Events
from ignite.trainer import Trainer
from ignite.evaluator import Evaluator
from ignite.handlers import Evaluate
from ignite.handlers.logging import log_simple_moving_average

from torchtext import data

from utils import load_yaml
from model import RNNClassifier, StackedCRNNClassifier
from hooks import (save_checkpoint_handler, restore_checkpoint_handler,
                   get_classification_report_handler)
from preprocessing import cleanup_text
from pydoc import locate

PARSER = argparse.ArgumentParser(
    description="Twitter Sentiment Analysis with char-rnn")
PARSER.add_argument(
    "--epochs", type=int, default=10000, help="Number of epochs")
PARSER.add_argument(
    "--dataset",
    type=str,
    default="./data/sentiment",
    help="""Path for your training, validation and test dataset.
    As this package uses torch text to load the data, please
    follow the format by providing the path and filename without its
    extension""")
PARSER.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="The number of batch size for every step")
PARSER.add_argument("--log_interval", type=int, default=100)
PARSER.add_argument("--save_interval", type=int, default=500)
PARSER.add_argument("--validation_interval", type=int, default=500)
PARSER.add_argument(
    "--model_config",
    type=str,
    default="config/rnn.yml",
    help="Location of model config")
PARSER.add_argument(
    "--model_dir", type=str, default="", help="Location to save the model")
ARGS = PARSER.parse_args()

if __name__ == "__main__":
    # Load necessary configs
    model_config = load_yaml(ARGS.model_config)
    device = -1  # Use CPU as a default device

    # Preparing seed
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        device = None  # Use GPU when available

    # Preparing dataset
    # Get dataset name
    dataset_path = '/'.join(ARGS.dataset.split("/")[:-1])
    dataset_name = ARGS.dataset.split("/")[-1]
    text = data.Field(
        preprocessing=cleanup_text,
        include_lengths=True,
        tokenize=lambda s: list(s))
    sentiment = data.Field(pad_token=None, unk_token=None)
    train, val = data.TabularDataset.splits(
        dataset_path,
        train=dataset_name + ".train",
        validation=dataset_name + ".val",
        format="csv",
        fields=[("sentiment", sentiment), ("text", text)])
    text.build_vocab(train.text, min_freq=1, max_size=80000)
    sentiment.build_vocab(train.sentiment)
    train_iter, val_iter = data.BucketIterator.splits(  # pylint: disable=E0632
        datasets=[train, val],
        batch_size=ARGS.batch_size,
        sort_key=lambda x: len(x.text),
        device=device,
        repeat=False)

    # Build model graph
    classifier = locate(model_config["model"])(
        config=model_config,
        vocab_size=len(text.vocab.itos),
        label_size=len(sentiment.vocab.itos))
    classifier = DataParallel(classifier)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters())

    def training_update_function(batch):
        classifier.train()
        optimizer.zero_grad()
        text, y = batch.text, batch.sentiment
        x = text[0]
        # seq_len must be in descending order
        seq_len = text[1].numpy()
        seq_len[::-1].sort()
        y_pred = classifier(x, seq_len)
        loss = loss_fn(y_pred, y.squeeze())
        loss.backward()
        optimizer.step()
        return loss.data.cpu()[0]

    def inference_function(batch):
        classifier.eval()
        text, y = batch.text, batch.sentiment
        x = text[0]
        seq_len = text[1].numpy()
        seq_len[::-1].sort()
        softmax = nn.Softmax(dim=1)

        y_pred = classifier(x, seq_len)
        y_pred = softmax(y_pred)
        return y_pred.data.cpu(), y.data.squeeze().cpu()

    trainer = Trainer(training_update_function)
    evaluator = Evaluator(inference_function)

    # Put event handlers
    trainer.add_event_handler(Events.STARTED,
                              restore_checkpoint_handler(
                                  classifier, ARGS.model_dir))
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        log_simple_moving_average,
        window_size=10,
        should_log=
        lambda engine: engine.current_iteration % ARGS.log_interval == 0,
        metric_name="CrossEntropy",
        logger=print)
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        save_checkpoint_handler(classifier, ARGS.model_dir),
        should_save=
        lambda engine: engine.current_iteration % ARGS.save_interval == 0)
    trainer.add_event_handler(Events.COMPLETED,
                              save_checkpoint_handler(classifier,
                                                      ARGS.model_dir))
    trainer.add_event_handler(Events.ITERATION_COMPLETED,
                              Evaluate(
                                  evaluator,
                                  val_iter,
                                  iteration_interval=ARGS.validation_interval))

    evaluator.add_event_handler(Events.COMPLETED,
                                get_classification_report_handler())

    # Start training
    trainer.run(train_iter, max_epochs=ARGS.epochs)
