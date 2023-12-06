from torchmetrics.classification import MultilabelPrecision, MultilabelF1Score, MultilabelAccuracy, MultilabelRecall


def classification_metrics(predictions, labels):
    precs = MultilabelPrecision(num_labels=1317)
    f1 = MultilabelF1Score(num_labels=1317)
    acc = MultilabelAccuracy(num_labels=1317)
    recall = MultilabelRecall(num_labels=1317)

    precision = precs(predictions, labels)
    f1_score = f1(predictions, labels)
    label_accuracy = acc(predictions, labels)
    rec = recall(predictions, labels)

    return precision, f1_score, label_accuracy, rec
