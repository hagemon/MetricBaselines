import torch
import torch.nn as nn


def distance(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class LiftedLoss(nn.Module):
    def __init__(self, margin):
        self.margin = margin
        super(LiftedLoss, self).__init__()

    def forward(self, embeddings, labels):
        # dist = embeddings.mm(embeddings.t())
        dist = distance(embeddings)
        n_classes = labels.unique()

        loss = 0.0
        count = 0

        for i, c in enumerate(n_classes):
            pos_mask = labels == c
            neg_mask = ~pos_mask
            ap = dist[pos_mask][:, pos_mask]
            an = dist[pos_mask][:, neg_mask]
            pn_loss = (self.margin-an).exp().sum(1)
            loss_val = torch.log(pn_loss.unsqueeze(1)+pn_loss.unsqueeze(0)).add(ap)
            mask = torch.eye(len(ap)).byte().to(embeddings.device) & torch.gt(loss_val, 0)
            loss += torch.masked_select(loss_val, mask).pow(2).sum()

            count += 2*len(ap)
        return loss, count
