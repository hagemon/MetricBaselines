import torch
import torch.nn as nn


class RankedListLoss(nn.Module):

    def __init__(self, margin, alpha):
        super(RankedListLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.t = 10.

    def forward(self, embeddings, labels):
        loss = 0.0
        n_classes = torch.unique(labels)
        dist = (embeddings.unsqueeze(1)-embeddings.unsqueeze(0)).pow(2).sum(2)
        for c in n_classes:
            pos_mask = labels == c
            neg_mask = ~pos_mask
            ap = dist[pos_mask][:, pos_mask]
            an = dist[pos_mask][:, neg_mask]
            ap = ap[(ap > (self.alpha-self.margin)) & ~torch.eye(len(ap)).byte().to(embeddings.device)]
            an = an[an < self.alpha]
            w = torch.exp(self.t*(self.alpha-an))
            w_sum = w.sum()
            if len(ap) > 0:
                pos_loss = (ap - self.alpha + self.margin).sum()
                loss += pos_loss
            if len(an) > 0:
                neg_loss = (self.alpha - an).mul(w.div(w_sum)).sum()
                loss += neg_loss
        return loss, len(embeddings)
