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
        return 1 - dice_coef(pred, target)


def dice_coef(pred, target):
    N, C, _ = target.shape
    smooth = 1
    total_loss = 0.
    total_loss = torch.tensor(0)
    for batch_i in range(N):
        for c in range(C):
            intersection = torch.dot(pred[batch_i, c, :], target[batch_i, c, :])

            loss = 2 * (intersection + smooth) \
                   / (pred.sum() + target.sum() + smooth)
            print("*****************************")
            print(total_loss, loss)
            print("*****************************")
            total_loss += loss
        total_loss = total_loss / C
    total_loss = total_loss / N
    return total_loss
