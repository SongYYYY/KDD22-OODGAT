import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, average_precision_score

def get_acc(y_true, y_pred, mask):
    n_correct = (y_true[mask] == y_pred[mask]).sum()
    n_total = mask.sum()

    return n_correct / n_total

# ood detection performance
def get_ood_performance(detection_y, ood_scores, detection_mask):
    '''
    compute performance for ood detection.
    :param detection_y: binary, 1 indicates ood.
    :param ood_scores: higher means more likely to be ood.
    :param detection_mask: mask for ood detection task.
    :return:
    auroc, fprs of [fpr@80, fpr@90, fpr@95]
    '''
    auroc = roc_auc_score(detection_y, ood_scores[detection_mask])
    aupr_0 = average_precision_score(detection_y, -1.0 * ood_scores[detection_mask], pos_label=0)
    aupr_1 = average_precision_score(detection_y, ood_scores[detection_mask], pos_label=1)
    fpr, tpr, thresholds = roc_curve(detection_y, ood_scores[detection_mask], drop_intermediate=False)
    fprs = []
    for p in [0.8, 0.9, 0.95]:
        f = fpr[abs((tpr - p))<0.005].mean()
        if not np.isnan(f):
            fprs.append(f)
        else:
            fprs.append(0)

    return auroc, aupr_0, aupr_1, fprs


# get micro, macro and weighted F-1 score for joint classification.
def get_f1_score(y_true, y_pred, ood_score, score_type, mask, n_id_classes):
    '''
    evaluate the F1 scores of N+1 clf task.
    parameters:
    score_type: 'ent' or 'att'.
    return:
    list of {micro, macro, weighted} on various thresholds.
    '''
    if score_type == 'ent':
        thresholds = list(np.arange(np.log(n_id_classes), step=0.01))[1:]
    elif score_type == 'att':
        thresholds = list(np.arange(1, step=0.01))[1:]

    scores = []
    ood_score = ood_score[mask]
    for t in thresholds:
        y_pred_t = y_pred[mask]
        y_pred_t[ood_score>t] = n_id_classes
        f1_micro = f1_score(y_true, y_pred_t, average='micro')
        f1_macro = f1_score(y_true, y_pred_t, average='macro')
        f1_weighted = f1_score(y_true, y_pred_t, average='weighted')

        scores.append({'micro': f1_micro, 'macro': f1_macro, 'weighted': f1_weighted, 'threshold': t})

    return scores




