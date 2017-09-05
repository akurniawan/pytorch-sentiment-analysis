import csv


def get_vocabs(filename):
    vocabs = {}
    categories = {}
    print("Load vocabs and categories")
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for i, r in enumerate(reader):
            for word in r[0]:
                if not word in vocabs:
                    vocabs[word] = len(vocabs) + 1
            print(r)
            if not r[1] in categories:
                categories[r[1]] = len(categories)

    vocabs[''] = 0
    print("Vocabs:")
    print(vocabs)
    print("Categories:")
    print(categories)
    print("Vocabs and categories are loaded!")
    return vocabs, categories
