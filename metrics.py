import torch


def accuracy(logits, labels):
    """Calculate accuracy for the current logits
    Args:
        logits: autograd.Variable with size of [batch x nlabels]
        labels: autograd.Variable with size of [batch]

    Return:
        type(double) of current accuracy
    """

    pred, pred_idx = torch.max(logits, 1)
    correct_predictions = (
        pred_idx.data == labels.data).sum()
    acc = correct_predictions / labels.size()[0]

    return acc