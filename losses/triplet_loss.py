import torch
import torch.nn as nn


def distance(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class TripletLoss(nn.Module):
    def __init__(self, margin):
        self.margin = margin
        super(TripletLoss, self).__init__()

    def forward(self, embeddings, labels):

        dist = distance(embeddings)
        unique_labels = labels.unique()
        losses = 0.0
        count = 0.0

        for label in unique_labels:
            pos_mask = labels == label
            neg_mask = ~pos_mask

            ap = dist[pos_mask][:, pos_mask]
            an = dist[pos_mask][:, neg_mask]

            loss = ap.unsqueeze(2)-an.unsqueeze(1)+self.margin
            mask = (loss > 0) & (loss < self.margin)
            loss = loss.masked_select(mask)
            losses += loss.sum()
            count += mask.sum()
        return losses, float(count)
