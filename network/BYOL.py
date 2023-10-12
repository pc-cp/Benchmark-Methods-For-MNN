from network.base_model import ModelBase_ResNet18
from network.heads import (
    BYOLPredictionHead, BYOLProjectionHead
)
import copy

from util.utils import *
class BYOL(nn.Module):
    def __init__(self, dim=128, dataset='cifar10', momentum=-1, bn_splits=8, symmetric=False):
        super(BYOL, self).__init__()

        self.momentum = momentum
        self.symmetric = symmetric
        # create the encoders
        self.net       = ModelBase_ResNet18(dataset=dataset, bn_splits=bn_splits)
        self.backbone_momentum = copy.deepcopy(self.net)

        self.projection_head = BYOLProjectionHead(input_dim=512, hidden_dim=2048, output_dim=dim)
        self.prediction_head = BYOLPredictionHead(input_dim=dim, hidden_dim=2048, output_dim=dim)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def contrastive_loss(self, im_q, im_k):

        # compute query features
        p_q = self.prediction_head(self.projection_head(self.net(im_q)))  # queries: NxC

        with torch.no_grad():  # no gradient to keys
            # shuffle
            im_k_, shuffle = batch_shuffle(im_k)
            z_k = self.projection_head_momentum(self.backbone_momentum(im_k_)).clone().detach()  # keys: NxC
            # undo shuffle
            z_k = batch_unshuffle(z_k, shuffle)

        q = nn.functional.normalize(p_q, dim=1)
        k = nn.functional.normalize(z_k, dim=1)

        dist_qk = 2 - 2 * torch.einsum('bc,kc->bk', [q, k])
        dist_qk = torch.diag(dist_qk)
        loss = dist_qk.mean()

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
        update_momentum(model=self.net,             model_ema=self.backbone_momentum,        m=self.momentum)
        update_momentum(model=self.projection_head, model_ema=self.projection_head_momentum, m=self.momentum)

        loss_12 = self.contrastive_loss(im1, im2)
        loss = loss_12

        # compute loss
        if self.symmetric:  # symmetric loss
            loss_21 = self.contrastive_loss(im2, im1)
            loss = (loss_12 + loss_21)*1.0/2

        return loss