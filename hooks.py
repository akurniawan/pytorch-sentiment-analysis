import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from pathlib import Path


def save_checkpoint_handler(model, model_path, logger=print):
    def save_checkpoint(trainer, should_save=lambda x: True):
        if should_save(trainer):
            try:
                logger("Saving model...")
                torch.save(model.state_dict(), model_path)
                logger("Finish saving model!")
            except Exception as e:
                logger("Something wrong while saving the model: %s" % str(e))

    return save_checkpoint


def restore_checkpoint_handler(model, model_path, logger=print):
    def restore_checkpoint(trainer):
        try:
            model_file = Path(model_path)
            if model_file.exists():
                logger("Start restore model...")
                model.load_state_dict(torch.load(model_path))
                logger("Finish restore model!")
            else:
                logger("Model not found, skip restoring model")
        except Exception as e:
            logger("Something wrong while restoring the model: %s" % str(e))

    return restore_checkpoint


def get_classification_report_handler(logger=print):
    def get_classification_report(evaluator):
        pred, y = zip(*evaluator.history)
        pred = torch.cat(pred, dim=0)
        y = torch.cat(y)

        y = y.numpy()
        _, indices = torch.max(pred, 1)
        indices = indices.numpy()

        print(indices.shape, y.shape)
        print("=====================================")
        print("Accuracy")
        print("=====================================")
        print("Accuracy Score: %f" % accuracy_score(y, indices))
        print("\n=====================================")
        print("Classifiation Report")
        print("=====================================")
        print(classification_report(y, indices))
        print("=====================================")
        print("Confusion Matrix")
        print("=====================================")
        print(confusion_matrix(y, indices))

    return get_classification_report