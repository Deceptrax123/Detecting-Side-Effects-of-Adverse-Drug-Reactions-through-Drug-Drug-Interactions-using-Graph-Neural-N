from torchmetrics.classification import MultilabelPrecision, MultilabelF1Score, MultilabelAccuracy, MultilabelRecall, MultilabelConfusionMatrix, MultilabelHammingDistance, MulticlassAveragePrecision
import numpy as np
import torch

# Precision@k


def topk_precision(predictions, labels, k):
    # sort the labels by predictions

    precisions = list()
    labs = list()
    scores = list()
    for i in range(predictions.size(0)):
        trues = 0
        values, indices = torch.sort(predictions[i], descending=True)

        if indices.size(0) >= k:
            # get top k predictions
            topk_pred_labels = indices[:k]
            topk_probs = values[:k]

            # search if top k labels are true or false classifications
            for j in topk_pred_labels:
                if labels[i][j.item()] == 1:
                    trues += 1

            precision = trues/k
        else:
            precision = 1

        precisions.append(precision)
        labs.append(topk_pred_labels)
        scores.append(torch.mean(topk_probs).item())

    return sum(precisions)/len(precisions), labs, sum(scores)/len(scores)


# Weighted Accruacy
def classification_metrics(predictions, labels):
    acc = MultilabelAccuracy(num_labels=1317, average='weighted')

    label_accuracy = acc(predictions, labels)

    return label_accuracy
