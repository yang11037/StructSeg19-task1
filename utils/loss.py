import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        """

        :param pred: （Batch, Classes, pixel）
        :param target:（Batch, Classes, pixel）
        :return: the mean loss of this batch
        """
        N, C, _ = target.shape
        smooth = 1
        total_loss = 0
        for batch_i in range(N):
            for c in range(C):
                intersection = torch.dot(pred[batch_i, c, :], target[batch_i, c, :])

                loss = 2 * (intersection + smooth) \
                       / (pred.sum(1) + target.sum(1) + smooth)
                total_loss += loss
            total_loss = total_loss / C
        total_loss = total_loss / N
        return total_loss
