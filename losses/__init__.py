from .lifted_loss import LiftedLoss
from .n_pair_loss import NPairLoss
from .triplet_loss import TripletLoss
from .ranked_list_loss import RankedListLoss
from .angular_loss import AngularLoss
import torch


def get_loss(method):
    losses = {
        'Triplet': TripletLoss(margin=1),
        'Angular': AngularLoss(),
        'RankedList': RankedListLoss(margin=0.4, alpha=1.2),
        'N_pair': NPairLoss(),
        'Lifted': LiftedLoss(margin=1)
    }
    return losses[method]


def class_centers(embeddings, labels):
    unique_classes = labels.unique()
    centers = torch.FloatTensor().to(embeddings.device)
    for c in unique_classes:
        indices = labels.eq(c)
        mean = embeddings[indices].mean(dim=0)
        centers = torch.cat((centers, mean.unsqueeze(0)), 0)
    return centers
