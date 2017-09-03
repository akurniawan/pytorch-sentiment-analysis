import torch
import linecache
import csv
import numpy as np

from vocab import get_vocabs
from operator import itemgetter
from collections import defaultdict

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

# TODO: put it on configuration
BUCKETS = [5, 25, 50, 100, 125, 175, 200, 225, 250, 300]

def get_bucket(sequence_length):
    bucket_sizes = [0] + BUCKETS + [np.inf]
    for i, bucket_size in enumerate(bucket_sizes):
        if sequence_length >= \
            bucket_size and sequence_length < bucket_sizes[i+1]:
            return bucket_sizes[i + 1]


def collate_fn(batch):
    labels = [i[1] for i in batch]
    sequences = [(i[0], len(i[0])) for i in batch]
    max_sequence = max(sequences, key=itemgetter(1))[1]
    sentence_buckets = defaultdict(list)
    category_buckets = defaultdict(list)
    seq_len_buckets = defaultdict(list)

    for i, sequence in enumerate(sequences):
        bucket_size = get_bucket(sequence[1])
        # print(bucket_size, sequence[1])
        if bucket_size == np.inf:
            num_pad = max_sequence - sequence[1]
        else:
            num_pad = bucket_size - sequence[1]
        sentence_buckets[bucket_size].append(
            np.concatenate([sequence[0],
                            np.zeros(num_pad, dtype=np.int8)]))
        category_buckets[bucket_size].append(labels[i])
        seq_len_buckets[bucket_size].append(sequence[1])

    for k, v in sentence_buckets.items():
        sentence_buckets[k] = np.array(v)
        sentence_buckets[k] = Variable(
            torch.LongTensor(sentence_buckets[k]), requires_grad=False)
        category_buckets[k] = Variable(
            torch.LongTensor(category_buckets[k]), requires_grad=False)
        seq_len_buckets[k] = torch.LongTensor(seq_len_buckets[k])

    return sentence_buckets, category_buckets, seq_len_buckets


class LazyTextDataset(Dataset):
    def __init__(self, filename):
        self._vocabs, self._categories = get_vocabs(filename)
        self._filename = filename
        self._total_data = 0
        with open(filename, "r") as f:
            self._total_data = len(f.readlines()) - 1

    def __getitem__(self, idx):
        # TODO: Better to just load all the data upfront? How to deal
        # with big data?
        line = linecache.getline(self._filename, idx + 1)
        csv_line = csv.reader([line])
        extracted_csv = [r for r in csv_line]
        if len(extracted_csv[0]) == 2:
            sentence, label = extracted_csv[0]
        else:
            print("Need to fix this data:", extracted_csv[0])
            sentence = "a"
            label = "0"
        entity_ids = []
        for entity in sentence:
            entity_ids.append(self._vocabs[entity])
        category_id = self._categories[label]

        return entity_ids, category_id

    def __len__(self):
        return self._total_data

    @property
    def vocab_size(self):
        return len(self._vocabs)

    @property
    def category_size(self):
        return len(self._categories)