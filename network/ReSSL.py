from network.base_model import ModelBase_ResNet18
from network.heads import (
    MoCoProjectionHead
)
import torch.nn.functional as F
import copy
from util.MemoryBankModule import MemoryBankModule
from util.utils import *

class ReSSL(nn.Module):
    def __init__(self, dim=128, K=4096, momentum=-1, tem=0.05, dataset='cifar10', bn_splits=8, symmetric=False):
        super(ReSSL, self).__init__()
        self.K = K
        self.momentum = momentum
        self.tem = tem
        self.symmetric = symmetric
        # create the encoders
        self.net               = ModelBase_ResNet18(dataset=dataset, bn_splits=bn_splits)
        self.backbone_momentum = copy.deepcopy(self.net)

        self.projection_head = MoCoProjectionHead(input_dim=512, hidden_dim=2048, output_dim=dim)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        self.memory_bank = MemoryBankModule(size=self.K).cuda()

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        # self.max_entropy = np.log(self.K)

    def contrastive_loss(self, im_q, im_k, labels, update=False):
        # compute query features
        z_q = self.projection_head(self.net(im_q))  # queries: NxC

        with torch.no_grad():  # no gradient to keys
            # shuffle
            im_k_, shuffle = batch_shuffle(im_k)
            z_k = self.projection_head_momentum(self.backbone_momentum(im_k_))  # keys: NxC
            # undo shuffle
            z_k = batch_unshuffle(z_k, shuffle)

        # Nearest Neighbour,    queue: [feature_dim, self.K]
        _, bank, _ = self.memory_bank(output=z_k, labels=labels, update=update)
        # ================normalized==================
        q = nn.functional.normalize(z_q, dim=1)
        k = nn.functional.normalize(z_k, dim=1)
        bank = nn.functional.normalize(bank.t(), dim=1)

        # calculate similiarities
        logits_q = torch.einsum("nc,mc->nm", q, bank)
        logits_k = torch.einsum("nc,mc->nm", k, bank)

        loss = - torch.sum(F.softmax(logits_k.detach() / self.tem, dim=1) * F.log_softmax(logits_q / 0.1, dim=1), dim=1).mean().cuda()
        return loss

    def forward(self, im1, im2, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # Updates parameters of `model_ema` with Exponential Moving Average of `model`
        update_momentum(model=self.net, model_ema=self.backbone_momentum, m=self.momentum)
        update_momentum(model=self.projection_head, model_ema=self.projection_head_momentum, m=self.momentum)

        loss_12 = self.contrastive_loss(im1, im2, update=True, labels=labels)
        loss = loss_12

        # compute loss
        if self.symmetric:  # symmetric loss
            loss_21 = self.contrastive_loss(im2, im1, update=False, labels=labels)
            loss = (loss_12 + loss_21) * 1.0 / 2
            # print('hi')

        return loss
