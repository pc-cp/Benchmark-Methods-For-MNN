import numpy as np
import torch.nn as nn
import torch
from network.base_model import ModelBase_ResNet18
from network.heads import (
    SimCLRProjectionHead
)
import torch.nn.functional as F
from util.utils import *
class SimCLR(nn.Module):
    def __init__(self, dim=128, tem=0.1, dataset='cifar10', bn_splits=8, symmetric=False):
        super(SimCLR, self).__init__()

        self.tem = tem
        self.symmetric = symmetric
        # create the encoders
        self.net       = ModelBase_ResNet18(dataset=dataset, bn_splits=bn_splits)
        self.projection_head = SimCLRProjectionHead(input_dim=512, hidden_dim=2048, output_dim=dim)
    def contrastive_loss(self, im_q, im_k):

        # compute query features
        z_q = self.projection_head(self.net(im_q))  # queries: NxC
        z_k = self.projection_head(self.net(im_k))

        batch_size, _ = z_q.shape
        # ================normalized==================
        z_q = nn.functional.normalize(z_q, dim=1)
        z_k = nn.functional.normalize(z_k, dim=1)  # already normalized

        # =============== standard implement simclr of lightly (except non-symmetric)================
        # diag_mask = torch.eye(batch_size, dtype=torch.bool).cuda()
        # calculate similiarities
        # logits_00 = torch.einsum("nc,mc->nm", p_q, p_q) / self.tem
        # logits_01 = torch.einsum("nc,mc->nm", p_q, z_k) / self.tem
        # logits_10 = torch.einsum("nc,mc->nm", z_k, p_q) / self.tem
        # logits_11 = torch.einsum("nc,mc->nm", z_k, z_k) / self.tem
        # remove simliarities between same views of the same image
        # logits_00 = logits_00[~diag_mask].view(batch_size, -1).cuda()
        # logits_11 = logits_11[~diag_mask].view(batch_size, -1).cuda()
        # concatenate logits
        # the logits tensor in the end has shape (2*n, 2*n-1)
        # logits_0100 = torch.cat([logits_01, logits_00], dim=1).cuda()
        # logits_1011 = torch.cat([logits_10, logits_11], dim=1).cuda()
        # logits = torch.cat([logits_0100, logits_1011], dim=0).cuda()

        # create labels
        # labels = torch.eye(batch_size).cuda()
        # labels = torch.cat([labels.repeat(2, 1), torch.zeros(2*batch_size, logits.shape[1]-(labels.repeat(2, 1).shape[1])).cuda()], dim=1).cuda()
        # =============== standard implement simclr of lightly================

        # ================= my implement(simclr) ==============
        logits = torch.einsum("nc,mc->nm", z_q, z_k) / self.tem
        labels = torch.eye(batch_size).cuda()

        loss = - torch.sum(labels * F.log_softmax(logits, dim=1), dim=1).mean().cuda()

        return loss

    def forward(self, im1, im2, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        loss_12 = self.contrastive_loss(im1, im2)
        loss = loss_12

        # compute loss
        if self.symmetric:  # symmetric loss
            loss_21 = self.contrastive_loss(im2, im1)
            loss = (loss_12 + loss_21)*1.0/2

        return loss