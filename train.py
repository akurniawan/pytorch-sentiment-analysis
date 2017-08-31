import dataset
import torch
import torch.nn as nn

from torch.autograd import Variable

from dataset import LazyTextDataset, DataLoader
from vocab import get_vocabs
from model import ClassifierModel

if __name__ == "__main__":
    vocabs, categories = get_vocabs("data/train.csv")
    train_dataset = LazyTextDataset("data/train.csv", vocabs, categories)
    train_data = DataLoader(
        dataset=train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn)
    classifier_model = ClassifierModel(100, 1, len(vocabs), 128, 57)
    loss_fn = nn.CrossEntropyLoss()

    for data_bucket, category_bucket in train_data:
        for data, category in zip(data_bucket.keys(), category_bucket.keys()):
            logits = classifier_model(Variable(data_bucket[data], requires_grad=False))
            # loss = loss_fn(logits, Variable(category_bucket[category], requires_grad=False))
            # loss.backward()
            # print(loss)
        # break