from network.base_model import ModelBase_ResNet18
from network.heads import (
    BYOLPredictionHead, BYOLProjectionHead
)
import copy
from util.MemoryBankModule import MemoryBankModule
from util.NNMemoryBankModule import NNMemoryBankModule
from util.utils import *
class CMSF(nn.Module):
    def __init__(self, dim=128, K=4096, momentum=-1, topk=1, dataset='cifar10', bn_splits=8, symmetric=False):
        super(CMSF, self).__init__()

        self.K = K
        self.momentum = momentum
        self.topk = topk
        self.symmetric = symmetric
        self.dim = dim
        # create the encoders
        self.net               = ModelBase_ResNet18(dataset=dataset, bn_splits=bn_splits)
        self.backbone_momentum = copy.deepcopy(self.net)

        self.projection_head = BYOLProjectionHead(input_dim=512, hidden_dim=2048, output_dim=dim)
        self.prediction_head = BYOLPredictionHead(input_dim=dim, hidden_dim=2048, output_dim=dim)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        self.memory_bank = NNMemoryBankModule(size=self.K, topk=self.topk).cuda()
        self.memory_bank_auxiliary = MemoryBankModule(size=self.K).cuda()

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def generation_mask_with_pos(self, batch_size):
        nn_labels = generation_mask(batch=batch_size, topk=self.topk).cuda()

        # ============insert one-hot encoding===========
        one_label = torch.eye(batch_size).cuda()
        labels = torch.cat((one_label.reshape(batch_size, -1, batch_size).unsqueeze(dim=-1),
                            nn_labels.reshape(batch_size, -1, batch_size, self.topk)),
                           dim=-1).reshape(batch_size, -1)

        return labels

    def nn_in_bank_auxiliary(self, z_v, bank_auxiliary, bank):
        z_v_normalized = nn.functional.normalize(z_v, dim=1)
        bank_auxiliary_normalized = nn.functional.normalize(bank_auxiliary.t(), dim=1)

        similarity_matrix = torch.einsum("nd,md->nm", z_v_normalized, bank_auxiliary_normalized)
        _, index_nearest_neighbours = similarity_matrix.topk(self.topk, dim=1, largest=True)
        nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours.reshape(-1))

        return nearest_neighbours

    def contrastive_loss(self, im_q, im_k, im_v, labels, update=False):

        # compute query features
        z_q = self.projection_head(self.net(im_q))  # queries: NxC
        p_q = self.prediction_head(z_q)

        with torch.no_grad():  # no gradient to keys
            # shuffle
            im_k_, shuffle = batch_shuffle(im_k)
            z_k = self.projection_head_momentum(self.backbone_momentum(im_k_)).clone().detach()  # keys: NxC
            # undo shuffle
            z_k = batch_unshuffle(z_k, shuffle)

            # shuffle
            im_v_, shuffle = batch_shuffle(im_v)
            z_v = self.projection_head_momentum(self.backbone_momentum(im_v_)).clone().detach()  # keys: NxC
            # undo shuffle
            z_v = batch_unshuffle(z_v, shuffle)

        batch_size, feature_dim = z_q.shape
        # Nearest Neighbour
        _, bank_auxiliary, _ = self.memory_bank_auxiliary(output=z_v, labels=labels, update=update)
        _, purity, bank, z_k_keep_nn = self.memory_bank(querys=z_k, keys=z_k, update=update, labels=labels)
        nn_auxiliary = self.nn_in_bank_auxiliary(z_v, bank_auxiliary, bank)
        z_k_keep_nn_auxiliary = torch.cat((z_k.unsqueeze(dim=1),
                                                      nn_auxiliary.reshape(batch_size, self.topk, feature_dim)),
                                                     dim=1).reshape(-1, feature_dim)


        # ================normalized==================
        p_q_normalized = nn.functional.normalize(p_q, dim=1)
        z_k_keep_nn_normalized = nn.functional.normalize(z_k_keep_nn, dim=1)
        z_k_keep_nn_auxiliary_normalized = nn.functional.normalize(z_k_keep_nn_auxiliary, dim=1)
        # shape (batch_size, batch_size*(1+topk))
        dist_qk_keep_nn = 2 - 2 * torch.einsum('bc,kc->bk', [p_q_normalized, z_k_keep_nn_normalized])
        dist_qk_keep_auxiliary = 2 - 2 * torch.einsum('bc,kc->bk', [p_q_normalized, z_k_keep_nn_auxiliary_normalized])
        pseudo_labels = self.generation_mask_with_pos(batch_size).cuda()

        loss = (torch.mul(dist_qk_keep_nn, pseudo_labels).sum(dim=1)/(1+self.topk)).mean()
        loss_auxiliary = (torch.mul(dist_qk_keep_auxiliary, pseudo_labels).sum(dim=1)/(1 + self.topk)).mean()
        return (loss+loss_auxiliary)/2, purity

    def forward(self, im1, im2, im3, labels):
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

        if self.symmetric:  # symmetric loss
            loss_21, purity_21 = self.contrastive_loss(im2, im1, im3, update=False, labels=labels)

        loss_12, purity_12 = self.contrastive_loss(im1, im2, im3, update=True, labels=labels)
        purity = purity_12
        loss = loss_12
        # compute loss
        if self.symmetric:  # symmetric loss
            # loss_21, purity_21 = self.contrastive_loss(im2, im1, im3, update=False, labels=labels)
            purity = (purity_12 + purity_21)/2
            loss = (loss_12 + loss_21)/2

        return loss, purity