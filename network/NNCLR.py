from network.base_model import ModelBase_ResNet18
from network.heads import (NNCLRPredictionHead,
    NNCLRProjectionHead,
)
from util.NNMemoryBankModule import NNMemoryBankModule
import torch.nn.functional as F
from util.utils import *
class NNCLR(nn.Module):
    def __init__(self, dim=128, K=4096, topk=1, tem=0.1, dataset='cifar10', bn_splits=8, symmetric=False):
        super(NNCLR, self).__init__()

        self.K = K
        self.topk = topk
        self.tem = tem
        self.symmetric = symmetric
        # create the encoders
        self.net       = ModelBase_ResNet18(dataset=dataset, bn_splits=bn_splits)

        # self.projection_head = NNCLRProjectionHead(input_dim=512, hidden_dim=512, output_dim=dim)
        # self.prediction_head = NNCLRPredictionHead(input_dim=dim, hidden_dim=512, output_dim=dim)
        # new_stdout
        self.projection_head = NNCLRProjectionHead(input_dim=512, hidden_dim=2048, output_dim=dim)
        self.prediction_head = NNCLRPredictionHead(input_dim=dim, hidden_dim=2048, output_dim=dim)
        self.memory_bank = NNMemoryBankModule(size=self.K, topk=self.topk).cuda()

    # def generation_mask(self, topk, batch):
    #     """Pseudo label generation for the introduction of top-k nearest neighbor methods.
    #         This code was taken and adapted from here:
    #         ###
    #
    #         Args:
    #             topk:
    #                 Number of neighbors
    #             batch:
    #                 batch_size
    #
    #         Returns:
    #             mask:
    #                 Pseudo-labeling with dimensions [batch, batch*topk]
    #     Examples:
    #     >>> generation_mask(2, 3)
    #     Out[5]:
    #     tensor([[1., 1., 0., 0., 0., 0.],
    #             [0., 0., 1., 1., 0., 0.],
    #             [0., 0., 0., 0., 1., 1.]])
    #     """
    #     mask_ = torch.eye(batch)
    #     mask = mask_.repeat(topk, 1).reshape(topk, batch, -1).permute(2, 1, 0).reshape(batch, topk*batch)
    #     return mas
    def contrastive_loss(self, im_q, im_k, labels, update=False):

        # compute query features
        z_q = self.projection_head(self.net(im_q))  # queries: NxC
        p_q = self.prediction_head(z_q)

        z_k = self.projection_head(self.net(im_k))
        # Nearest Neighbour
        z_k_nn, purity, _, _ = self.memory_bank(querys=z_k, keys=z_k, update=update, labels=labels)

        p_q = nn.functional.normalize(p_q, dim=1)  # already normalized
        z_k_nn = nn.functional.normalize(z_k_nn, dim=1)  # already normalized

        batch_size, _ = z_q.shape

        # concatenate logits
        # the logits tensor in the end has shape (n, n*topk)
        logits_nn_1 = torch.einsum("nc,mc->nm", z_k_nn, p_q) / self.tem
        logits_nn_2 = torch.einsum("nc,mc->nm", p_q,    z_k_nn) / self.tem

        # create labels
        labels_nn = generation_mask(batch=batch_size, topk=self.topk).cuda()

        loss = - ((torch.sum(labels_nn.t() * F.log_softmax(logits_nn_1, dim=1), dim=1).mean()+torch.sum(labels_nn * F.log_softmax(logits_nn_2, dim=1), dim=1).mean())/2.0).cuda()

        return loss, purity

    def forward(self, im1, im2, labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        if self.symmetric:  # symmetric loss
            loss_21, purity_21 = self.contrastive_loss(im2, im1, update=False, labels=labels)

        loss_12, purity_12 = self.contrastive_loss(im1, im2, update=True, labels=labels)
        loss = loss_12
        purity = purity_12
        # compute loss
        if self.symmetric:  # symmetric loss
            # loss_21, purity_21 = self.contrastive_loss(im2, im1, update=False, labels=labels)
            purity = (purity_12 + purity_21)*1.0/2
            loss = (loss_12 + loss_21)*1.0/2

        return loss, purity