import torch
import torch.nn as nn
import numpy as np


class AngularLoss(nn.Module):
    """
    Angular loss
    Wang, Jian. "Deep Metric Learning with Angular Loss," CVPR, 2017
    https://arxiv.org/pdf/1708.01682.pdf
    """

    def __init__(self, l2_reg=0.02, angle_bound=1., lambda_ang=2, n_pair=False):
        super(AngularLoss, self).__init__()
        self.l2_reg = l2_reg
        self.angle_bound = angle_bound
        self.lambda_ang = lambda_ang
        self.n_pair = n_pair

    def forward(self, embeddings, labels):
        n_pairs, n_negatives = self.get_n_pairs(labels)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]  # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]  # (n, n-1, embedding_size)

        losses = self.angular_loss(
            anchors, positives, negatives
        ).add(
            self.l2_loss(anchors, positives)*self.l2_reg
        )
        count = len(anchors)

        return losses, count

    def angular_loss(self, anchors, positives, negatives):
        """
        Calculates angular loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(
            (anchors.add(positives)),
            negatives.transpose(1, 2)
        ).mul(4*self.angle_bound).sub(
            torch.matmul(
                anchors,
                positives.transpose(1, 2)
            ).mul(2.*(1.+self.angle_bound))
        )

        # Preventing overflow
        with torch.no_grad():
            t = torch.max(x, dim=2)[0]

        x = torch.exp(x.sub(t.unsqueeze(dim=1)))
        x = torch.log(torch.exp(-t).add(torch.sum(x, 2)))
        loss = torch.sum(t.add(x))

        if self.n_pair:
            n_pair_loss = self.n_pair_loss(anchors, positives, negatives)
            loss = (loss.mul(self.lambda_ang).add(n_pair_loss)).div(self.lambda_ang+1.)

        return loss

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False)
            n_pairs.append([anchor, positive])

        n_pairs = np.array(n_pairs)

        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.concatenate([n_pairs[:i, 1], n_pairs[i + 1:, 1]])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return (anchors.pow(2) + positives.pow(2)).sum().div(float(anchors.size(0)))
