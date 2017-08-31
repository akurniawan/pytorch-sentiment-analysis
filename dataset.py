import linecache
import numpy as np

from operator import itemgetter
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from torch import from_numpy


def get_bucket(sequence_length):
    bucket_sizes = [0, 5, 25, 50, 100, 125, 175, 200, 225, 250, 300, np.inf]
    for i, bucket_size in enumerate(bucket_sizes):
        if sequence_length >= bucket_size and sequence_length < bucket_sizes[i+1]:
            return bucket_sizes[i+1]


def collate_fn(batch):
    labels = [i[1] for i in batch]
    sequences = [(i[0], len(i[0])) for i in batch]
    max_sequence = max(sequences, key=itemgetter(1))[1]
    sentence_buckets = defaultdict(list)
    category_buckets = defaultdict(list)

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

    for k, v in sentence_buckets.items():
        sentence_buckets[k] = np.array(v)
        sentence_buckets[k] = from_numpy(sentence_buckets[k])
        category_buckets[k] = np.array(category_buckets[k])
        category_buckets[k] = from_numpy(category_buckets[k])

    return sentence_buckets, category_buckets


class LazyTextDataset(Dataset):
    def __init__(self, filename, vocabs, categories):
        self._vocabs = vocabs
        self._categories = categories
        self._filename = filename
        self._total_data = 0
        with open(filename, "r") as f:
            self._total_data = len(f.readlines()) - 1

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        csv_line = csv.reader([line])
        sentence, label = [r for r in csv_line][0]
        sentence = sentence.split()
        word_ids = []
        for word in sentence:
            word_ids.append(self._vocabs[word])
        category_id = self._categories[label]

        return word_ids, category_id

    def __len__(self):
        return self._total_data

