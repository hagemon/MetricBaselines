import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score


def test(filename, test_ft, test_labels):
    """
    Test model with NMI, F1 and R@radius
    :param filename: filename with method and dataset
    :param test_ft: features of test set
    :param test_labels: labels corresponding to features
    :return: None
    """
    metrics_path = os.path.join('metrics')
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path, exist_ok=True)
    metrics = {}
    # Recall @ radius
    radius = [1, 2, 4, 8]
    r_res = recall_r(test_ft, test_labels, radius)
    for i in range(len(radius)):
        print('R@{}: {}'.format(radius[i], r_res[i]))
        metrics['R@{}'.format(radius[i])] = r_res[i]

    # NMI and F1
    nmi_res, f1_res = nmi_f1(test_ft, test_labels)
    print('NMI: {}'.format(nmi_res))
    metrics['NMI'] = nmi_res
    print('F1: {}'.format(f1_res))
    metrics['F1'] = f1_res

    np.save(os.path.join(metrics_path, filename+'.npy'), metrics)


def dist_func(a, b):
    dist = -2 * np.matmul(a, np.transpose(b)) + np.sum(np.square(a), 1).reshape(1, -1) + np.sum(
        np.square(b), 1).reshape(-1, 1)
    return dist


def recall_r(ft, labels, radius):
    """
    Computed the percentage of test samples which have at least one example
    from the same category in R nearest neighbors for retrieval problem.

    Note that value of labels do not match indices,
    i.e. for CUB, labels varies from 100 to 200 in test set,
    but using a [n_samples, 200] matrix do not affect result for recall@r metrics,
    because we calculate whether two samples share a label.

    :param ft: features of test set.
    :param labels: labels of test set.
    :param radius: list type, w.r.t different radius needed in testing procedure.
    :return: percentage that meets requirement in various radius.
    """
    n_classes = max(labels) + 1
    labels_mat = np.zeros([len(labels), n_classes], dtype=np.int)
    labels_mat[np.arange(len(labels)), labels] = 1  # one-hot matrix

    dist = dist_func(ft, ft)
    max_radius = max(radius)
    indices = dist.argsort()[:, 1:max_radius+1]  # zero dist to self, so start from 1
    hit = np.sum(labels_mat[indices] * np.expand_dims(labels_mat, 1), axis=2)
    res = np.zeros_like(radius, dtype=np.float)
    for i in range(len(radius)):
        res[i] = np.mean(np.sum(hit[:, :radius[i]], 1).astype(bool))
    return res


def nmi_f1(ft, labels):
    n_samples = len(ft)
    unique_labels = np.unique(labels)
    n_clusters = unique_labels.shape[0]
    k_means = KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=0).fit(ft)
    # NMI
    nmi_res = normalized_mutual_info_score(labels, k_means.labels_, average_method='geometric')
    # F1 score
    unique_centers = np.unique(k_means.labels_)
    matrix = np.zeros([len(unique_labels), len(unique_centers)])
    weight = np.zeros_like(unique_labels, dtype=float)
    for i in range(n_clusters):
        label = unique_labels[i]
        weight[i] = np.sum(labels == label).astype(float) / n_samples
        for j in range(n_clusters):
            center = unique_centers[j]
            label_ind = labels == label
            center_ind = k_means.labels_ == center
            inter = np.sum(label_ind*center_ind).astype(float)
            prec = inter / np.sum(center_ind, dtype=float)
            recall = inter / np.sum(label_ind, dtype=float)
            if prec + recall == 0:
                continue
            f = 2*prec*recall/(prec+recall)
            matrix[i, j] = f
    f1_res = np.max(matrix, 1)
    f1_res = np.sum(f1_res * weight)

    return nmi_res, f1_res







