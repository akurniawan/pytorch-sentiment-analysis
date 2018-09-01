import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

from ignite.engine import Events, Engine
from ignite.metrics import CategoricalAccuracy, Precision, Recall
from metrics import Loss

from torchtext import data

from utils import load_yaml
from model import RNNClassifier, StackedCRNNClassifier
from handlers import ModelLoader, ModelCheckpoint
from preprocessing import cleanup_text
from helper import create_supervised_evaluator
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
    default=16,
    help="The number of batch size for every step")
PARSER.add_argument("--log_interval", type=int, default=100)
PARSER.add_argument("--save_interval", type=int, default=500)
PARSER.add_argument("--validation_interval", type=int, default=500)
PARSER.add_argument(
    "--char_level",
    help="Whether to use the model with "
    "character level or word level embedding. Specify the option "
    "if you want to use character level embedding")
PARSER.add_argument(
    "--model_config",
    type=str,
    default="config/rnn.yml",
    help="Location of model config")
PARSER.add_argument(
    "--model_dir",
    type=str,
    default="models",
    help="Location to save the model")
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

    if ARGS.char_level:
        tokenize = lambda s: list(s)
    else:
        tokenize = lambda s: s.split()
    # Preparing dataset
    # Get dataset name
    dataset_path = '/'.join(ARGS.dataset.split("/")[:-1])
    dataset_name = ARGS.dataset.split("/")[-1]
    text = data.Field(
        preprocessing=cleanup_text, include_lengths=True, tokenize=tokenize)
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

    def training_update_function(engine, batch):
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
        return loss.cpu()

    def inference_function(engine, batch):
        classifier.eval()
        text, y = batch.text, batch.sentiment
        x = text[0]
        seq_len = text[1].numpy()
        seq_len[::-1].sort()
        softmax = nn.Softmax(dim=1)

        y_pred = classifier(x, seq_len)
        y_pred = softmax(y_pred)
        return y_pred.cpu(), y.squeeze().cpu()

    trainer = Engine(training_update_function)
    evaluator = create_supervised_evaluator(
        model=classifier,
        inference_fn=inference_function,
        metrics={
            "loss": Loss(loss_fn),
            "acc": CategoricalAccuracy(),
            "prec": Precision(),
            "rec": Recall()
        })
    checkpoint = ModelCheckpoint(
        ARGS.model_dir,
        "sentiment",
        save_interval=ARGS.save_interval,
        n_saved=5,
        create_dir=True,
        require_empty=False)
    loader = ModelLoader(classifier, ARGS.model_dir, "sentiment")
    model_name = model_config["model"].split(".")[1]

    # Event handlers
    trainer.add_event_handler(Events.STARTED, loader, model_name)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, checkpoint,
                              {model_name: classifier.module})
    trainer.add_event_handler(Events.COMPLETED, checkpoint,
                              {model_name: classifier.module})

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        if engine.state.iteration % ARGS.log_interval == 0:
            iterations_per_epoch = len(engine.state.dataloader)
            current_iteration = engine.state.iteration % iterations_per_epoch
            if current_iteration == 0:
                current_iteration = iterations_per_epoch
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  "".format(engine.state.epoch, current_iteration,
                            iterations_per_epoch, engine.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_iter)
        metrics = evaluator.state.metrics
        avg_loss = metrics["loss"]
        avg_accuracy = metrics["acc"]
        print("=====================================")
        print("Validation Results - Epoch: {}".format(engine.state.epoch))
        print("Avg accuracy: {:.2f}\nAvg loss: {:.2f}".format(
            avg_accuracy, avg_loss))
        print("Precision: {}".format(metrics["prec"].cpu()))
        print("Recall: {}".format(metrics["rec"].cpu()))
        print("=====================================")

    # Start training
    trainer.run(train_iter, max_epochs=ARGS.epochs)
