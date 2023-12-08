from torchmetrics.classification import MultilabelPrecision, MultilabelF1Score, MultilabelAccuracy, MultilabelRecall, MultilabelConfusionMatrix, MultilabelHammingDistance, MulticlassAveragePrecision


def classification_metrics(predictions, labels):
    precs = MultilabelPrecision(num_labels=1317, average='weighted')
    f1 = MultilabelF1Score(num_labels=1317, average='weighted')
    acc = MultilabelAccuracy(num_labels=1317, average='weighted')
    recall = MultilabelRecall(num_labels=1317, average='weighted')
    hamming = MultilabelHammingDistance(num_labels=1317, average='weighted')
    average_prec = MultilabelPrecision(num_labels=1317, average='weighted')

    # matrix = MultilabelConfusionMatrix(num_labels=1317)

    precision = precs(predictions, labels)
    f1_score = f1(predictions, labels)
    label_accuracy = acc(predictions, labels)
    rec = recall(predictions, labels)
    hamm = hamming(predictions, labels)
    avg_prec = average_prec(predictions, labels)

    return precision, f1_score, label_accuracy, rec, hamm, avg_prec
