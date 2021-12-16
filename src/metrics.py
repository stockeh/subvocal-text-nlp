import numpy as np


def confusion(Y, T, classes=None):
    if classes is None:
        classes = np.unique(T)
    confmat = np.zeros(
        (len(classes), len(classes)), dtype=int)
    for i in range(len(T)):
        confmat[T[i], Y[i]] += 1
    return confmat


def evaluate(Y, T, verbose=True):
    '''Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    confmat = confusion(Y, T, np.unique(T))

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.diag(confmat) / \
            np.sum(confmat, axis=0)  # tp / (tp + fp)
        recall = np.diag(confmat) / \
            np.sum(confmat, axis=1)  # tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)  # per class

    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)

    accuracy = np.trace(confmat) / len(T)

    if verbose:
        print_metrics(confmat, precision, recall, f1, accuracy, np.unique(T))

    return confmat, precision, recall, f1, accuracy


def print_confmat(confmat, class_names=None):
    classes = np.arange(
        confmat.shape[0]) if class_names is None else class_names
    spacing = len(str(np.max(confmat)))
    class_spacing = len(str(np.max(classes)))+1
    if class_spacing > spacing:
        spacing = class_spacing
    top = ' '*(class_spacing) + ''.join(' {i: < {spacing}}'.format(
        i=i, spacing=str(spacing)) for i in classes)
    t = ['{c:<{spacing}} |'.format(c=classes[j], spacing=str(spacing-1)) + ''.join(' {i:<{spacing}}'.format(
        i=i, spacing=str(spacing)) for i in row) for j, row in enumerate(confmat)]
    hdr = ' '*class_spacing + '-'*(len(t[0]) - class_spacing)
    print('Confusion Matrix:', top, hdr, '\n'.join(t), sep='\n')


def print_metrics(confmat, precision, recall, f1, accuracy, class_names):
    # Print Classes
    wrap = '-'*40
    print(wrap)
    # print('Classes:', ', '.join(f'{k}: {v}' for k,
    #                             v in self.class_dict.items()), end='\n\n')
    # Print Confusion Matrix
    print_confmat(confmat, class_names=class_names)

    # All-Class Metrics
    labels = ['Precision', 'Recall', 'F1']
    precision = np.append(precision, precision.mean())
    recall = np.append(recall, recall.mean())
    f1 = np.append(f1, 2*precision.mean()*recall.mean() /
                   (precision.mean()+recall.mean()))
    # Print Metrics
    metrics = np.vstack([precision, recall, f1])
    label_spacing = max([len(l) for l in labels])+1
    metric_spacing = max([len(f'{m:.3f}') for m in metrics.flatten()])
    mean = '  mean'
    top = ' '*(label_spacing) + ''.join(' {i: < {spacing}}'.format(
        i=i, spacing=str(metric_spacing)) for i in class_names) + mean
    t = ['{i:<{spacing}}|'.format(i=labels[j], spacing=str(label_spacing)) + ''.join(f' {i:.3f}' for i in row)
         for j, row in enumerate(metrics)]
    hdr = ' '*label_spacing + '-'*(len(t[0]) - label_spacing)
    print('\nMetrics:', top, hdr, '\n'.join(t), sep='\n')
    # Print Accuracy
    print(f'\nOverall Accuracy: {accuracy*100:.3f} %')
    print(wrap)
