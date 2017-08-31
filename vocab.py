import csv


def get_vocabs(filename):
    vocabs = {}
    categories = {}
    print("Start loading vocabs")
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for i, r in enumerate(reader):
            if i > 0:
                for word in r[0].split(" "):
                    if not word in vocabs:
                        vocabs[word] = len(vocabs) + 1
                if not r[1] in categories:
                    categories[r[1]] = len(categories) + 1

    vocabs[''] = 0
    print("Finish loading vocabs")
    return vocabs, categories
