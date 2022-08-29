import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import random

class EntropyLoss(nn.Module):
    '''
    return: mean entropy of the given batch if reduction is True, n-dim vector of entropy if reduction is False.
    '''
    def __init__(self, reduction=True):
        super(EntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        if self.reduction:
            b = -1.0 * b.sum()
            b = b / x.shape[0]
        else:
            b = -1.0 * b.sum(axis=1)

        return b

def cosine_similarity(x1, x2, reduction=True):
    '''
    compute cosine similarity between x1 and x2.
    :param x1: N * D tensor or 1d tensor.
    :param x2: N * D tensor or 1d tensor.
    :return: a scalar tensor if reduction is True, a tensor of shape N if reduction is False.
    '''
    cos_sim = nn.CosineSimilarity(dim=-1)
    if reduction:
        sim = cos_sim(x1, x2).mean()
    else:
        sim = cos_sim(x1, x2)

    return sim


class CE_uniform(nn.Module):
    '''
    return: CE of the given batch if reduction is True, n-dim vector of CE if reduction is False.
    '''
    def __init__(self, n_id_classes, reduction=True):
        super(CE_uniform, self).__init__()
        self.reduction = reduction
        self.n_id_classes = n_id_classes

    def forward(self, x):
        b = (1/self.n_id_classes) * F.log_softmax(x, dim=1)
        if self.reduction:
            b = -1.0 * b.sum()
            b = b / x.shape[0]
        else:
            b = -1.0 * b.sum(axis=1)

        return b


def get_consistent_loss_new(x1, x2, f1=None, f2=None):
    '''
    compute consistent loss between attention scores and output entropy.
    :param x1: ood score matrix, H * N tensor. the larger, the more likely to be ood.
    :param x2: entropy vector, N-dim tensor.
    :return: scalar tensor of computed loss.
    '''
    x1 = x1.mean(axis=0)
    if f1 is not None:
        x1 = f1(x1)
    if f2 is not None:
        x2 = f2(x2)
    loss = cosine_similarity(x1, x2)

    return -1.0 * loss

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    return