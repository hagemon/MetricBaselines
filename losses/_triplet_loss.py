import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations


def distance(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class TripletLoss(nn.Module):
    def __init__(self, margin):
        self.margin = margin
        super(TripletLoss, self).__init__()

    def semi_hard(self, loss):
        semi_hard_negatives = np.where(np.logical_and(loss < self.margin, loss > 0))[0]
        return np.random.choice(semi_hard_negatives) if len(semi_hard_negatives) > 0 else None

    def get_triplets(self, embeddings, labels):
        dist = distance(embeddings).cpu()
        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            pos = (labels == label)
            pos_inds = np.where(pos)[0]
            if len(pos_inds) < 2:
                continue
            neg_ind = np.where(np.logical_not(pos))[0]
            anchor_positives = list(combinations(pos_inds, 2))
            anchor_positives = np.array(anchor_positives)
            ap_distances = dist[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - dist[
                    anchor_positive[0],
                    neg_ind
                ] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.semi_hard(loss_values)
                if hard_negative is not None:
                    hard_negative = neg_ind[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

    def forward(self, embeddings, labels):

        triplets = self.get_triplets(embeddings.clone().detach(), labels)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.sum(), len(triplets)
