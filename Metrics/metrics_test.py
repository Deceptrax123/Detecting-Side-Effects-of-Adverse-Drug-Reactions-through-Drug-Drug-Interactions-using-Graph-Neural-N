# Metrics for Validation

from torchmetrics.classification import MultilabelPrecision, MultilabelF1Score, MultilabelAccuracy, MultilabelAUROC, MultilabelSpecificity, MultilabelRecall
import numpy as np
import torch


def classification_metrics(predictions, labels):
    acc = MultilabelAccuracy(num_labels=1317, average='weighted')
    f1 = MultilabelF1Score(num_labels=1317, average='micro')
    precision = MultilabelPrecision(num_labels=1317, average='micro')
    area = MultilabelAUROC(num_labels=1317, average='weighted')
    recall = MultilabelRecall(num_labels=1317, average='micro')

    label_accuracy = acc(predictions, labels)
    f1_micro = f1(predictions, labels)
    prec = precision(predictions, labels)
    auroc = area(predictions, labels)
    rec = recall(predictions, labels)

    return label_accuracy, f1_micro, prec, auroc, rec
