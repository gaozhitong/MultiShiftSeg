import numpy as np
import sklearn.metrics as sk
import time

"""
segmentation metrics
"""

# voc cityscapes metric
def hist_info(n_cl, pred, gt):
    assert (pred.shape == gt.shape)
    k = (gt >= 0) & (gt < n_cl)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == gt[k]))

    return np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                       minlength=n_cl ** 2).reshape(n_cl,
                                                    n_cl), labeled, correct


def compute_metric(results, per_class=False):
    hist = np.zeros((19, 19))
    correct = 0
    labeled = 0
    count = 0
    for d in results:
        try:
            hist += d['hist']
        except:
            import ipdb; ipdb.set_trace()
        correct += d['correct']
        labeled += d['labeled']
        count += 1
    if per_class:
        iu, mean_IU, class_acc, mean_pixel_acc = compute_score_per_class(hist, correct, labeled)
        return mean_IU, mean_pixel_acc, iu, class_acc
    else:
        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct, labeled)
        return mean_IU, mean_pixel_acc


def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled
    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc

def compute_score_per_class(hist, correct, labeled):
    # Intersection and Union
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    iu = intersection / np.maximum(union, 1)

    # Per-class accuracy
    class_acc = intersection / np.maximum(hist.sum(axis=1), 1)

    # Mean IoU and mean pixel accuracy
    mean_IU = np.nanmean(iu)
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, class_acc, mean_pixel_acc

"""
OOD metrics
"""
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level= 0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    del pos, neg, _pos, _neg


    auroc = 0.0
    # start = time.time()
    auroc = sk.roc_auc_score(labels, examples)
    # print("calculating auroc costs ", time.time()-start)


    aupr = sk.average_precision_score(labels, examples)



    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)


    return auroc, aupr, fpr


def get_and_print_results(out_score, in_score):
    aurocs, auprs, fprs = [], [], []

    measures = get_measures(out_score, in_score)
    aurocs.append(measures[0]);
    auprs.append(measures[1]);
    fprs.append(measures[2])

    auroc = np.mean(aurocs);
    aupr = np.mean(auprs);
    fpr = np.mean(fprs)

    return auroc, aupr, fpr

def eval_ood_measure(conf, seg_label, train_id_in=0, train_id_out=1):
    in_scores = conf[seg_label == train_id_in]
    out_scores = conf[seg_label == train_id_out]

    del seg_label, conf

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr
    else:
        return None
