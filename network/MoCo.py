from network.base_model import ModelBase_ResNet18
from network.heads import (
    MoCoProjectionHead
)
import copy
from util.MemoryBankModule import MemoryBankModule
from util.utils import *
class MoCo(nn.Module):
    def __init__(self, dim=128, K=4096, tem=0.1, momentum=0.99, dataset='cifar10', bn_splits=8, symmetric=False):
        super(MoCo, self).__init__()

        self.K = K
        self.tem = tem
        self.momentum = momentum
        self.symmetric = symmetric
        # create the encoders
        self.net               = ModelBase_ResNet18(dataset=dataset, bn_splits=bn_splits)
        self.backbone_momentum = copy.deepcopy(self.net)

        self.projection_head = MoCoProjectionHead(input_dim=512, hidden_dim=2048, output_dim=dim)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        self.memory_bank = MemoryBankModule(size=self.K).cuda()

    def contrastive_loss(self, im_q, im_k, labels, update=False):

        # compute query features
        z_q = self.projection_head(self.net(im_q))  # queries: NxC

        with torch.no_grad():  # no gradient to keys
            # shuffle
            im_k_, shuffle = batch_shuffle(im_k)
            z_k = self.projection_head_momentum(self.backbone_momentum(im_k_)).clone().detach()  # keys: NxC
            # undo shuffle
            z_k = batch_unshuffle(z_k, shuffle)

        # bank's shape: [feature_dim, batch_size]
        _, bank, _ = self.memory_bank(output=z_k, labels=labels, update=update)
        # ================normalized==================
        z_q = nn.functional.normalize(z_q, dim=1)
        bank = nn.functional.normalize(bank, dim=0)
        z_k = nn.functional.normalize(z_k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [z_q, z_k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [z_q, bank])
        # shape: batch_size * (1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = nn.CrossEntropyLoss().cuda()(logits/self.tem, labels)
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

        loss_12 = self.contrastive_loss(im1, im2, labels=labels, update=True)
        loss = loss_12

        # compute loss
        if self.symmetric:  # symmetric loss
            loss_21 = self.contrastive_loss(im2, im1,  labels=labels, update=False)
            loss = (loss_12 + loss_21)*1.0/2

        return loss