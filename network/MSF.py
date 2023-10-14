from network.base_model import ModelBase_ResNet18
from network.heads import (
    BYOLPredictionHead, BYOLProjectionHead
)
import copy
from util.NNMemoryBankModule import NNMemoryBankModule
from util.utils import *
class MSF(nn.Module):
    def __init__(self, dim=128, K=4096, momentum=-1, topk=1, dataset='cifar10', bn_splits=8, symmetric=False):
        super(MSF, self).__init__()

        self.K = K
        self.momentum = momentum
        self.topk = topk
        self.symmetric = symmetric
        # create the encoders
        self.net               = ModelBase_ResNet18(dataset=dataset, bn_splits=bn_splits)
        self.backbone_momentum = copy.deepcopy(self.net)

        self.projection_head = BYOLProjectionHead(input_dim=512, hidden_dim=2048, output_dim=dim)
        self.prediction_head = BYOLPredictionHead(input_dim=dim, hidden_dim=2048, output_dim=dim)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        self.memory_bank = NNMemoryBankModule(size=self.K, topk=self.topk).cuda()

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def _generation_mask(self, batch_size):
        '''
        generation_mask(3)
    Out[5]:
        tensor([[1., 1., 0., 0., 0., 0.],
                [0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 1., 1.]])
        '''
        mask_ = torch.eye(batch_size)
        mask = mask_.repeat(self.topk, 1).reshape(self.topk, batch_size, -1).permute(2, 1, 0).reshape(batch_size, self.topk*batch_size)

        return mask

    def generation_mask_with_pos(self, batch_size):
        nn_labels = generation_mask(batch=batch_size, topk=self.topk).cuda()

        # ============insert one-hot encoding===========
        one_label = torch.eye(batch_size).cuda()
        labels = torch.cat((one_label.reshape(batch_size, -1, batch_size).unsqueeze(dim=-1),
                            nn_labels.reshape(batch_size, -1, batch_size, self.topk)),
                           dim=-1).reshape(batch_size, -1)

        return labels

    def contrastive_loss(self, im_q, im_k, labels, update=False):

        # compute query features
        z_q = self.projection_head(self.net(im_q))  # queries: NxC
        p_q = self.prediction_head(z_q)

        with torch.no_grad():  # no gradient to keys
            # shuffle
            im_k_, shuffle = batch_shuffle(im_k)
            z_k = self.projection_head_momentum(self.backbone_momentum(im_k_)).clone().detach()  # keys: NxC
            # undo shuffle
            z_k = batch_unshuffle(z_k, shuffle)

        batch_size, _ = z_q.shape
        # Nearest Neighbour
        _, purity, _, z_k_keep_nn = self.memory_bank(querys=z_k, keys=z_k, update=update, labels=labels)

        # ================normalized==================
        p_q = nn.functional.normalize(p_q, dim=1)
        z_k_keep_nn = nn.functional.normalize(z_k_keep_nn, dim=1)

        # calculate distance between p_q and z_k_keep_nn, has shape (batch_size, batch_size*(1+topk))
        dist_qk_keep_nn = 2 - 2 * torch.einsum('bc,kc->bk', [p_q, z_k_keep_nn])
        pseudo_labels = self.generation_mask_with_pos(batch_size).cuda()

        loss = (torch.mul(dist_qk_keep_nn, pseudo_labels).sum(dim=1)/(1+self.topk)).mean()

        return loss, purity

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

        loss_12, purity_12 = self.contrastive_loss(im1, im2, update=True, labels=labels)
        loss = loss_12
        purity = purity_12
        # compute loss
        if self.symmetric:  # symmetric loss
            loss_21, purity_21 = self.contrastive_loss(im2, im1, update=False, labels=labels)
            purity = (purity_12 + purity_21)*1.0/2
            loss = (loss_12 + loss_21)*1.0/2

        return loss, purity