from network.base_model import ModelBase_ResNet18
import copy
from util.NNMemoryBankModule import NNMemoryBankModule
import torch.nn.functional as F
from util.utils import *
class SNCLR(nn.Module):
    def __init__(self, dim=128, K=4096, momentum=-1, topk=1, tem=0.1, dataset='cifar10', bn_splits=8, symmetric=False, threshold=False):
        super(SNCLR, self).__init__()

        self.K = K
        self.momentum = momentum
        self.topk = topk
        self.tem = tem
        self.threshold = threshold
        self.symmetric = symmetric
        # create the encoders
        self.net               = ModelBase_ResNet18(dataset=dataset, bn_splits=bn_splits)
        self.backbone_momentum = copy.deepcopy(self.net)

        self.projection_head = self._build_mlp(num_layers=3, input_dim=512, hidden_dim=2048, output_dim=dim)
        self.prediction_head = self._build_mlp(num_layers=2, input_dim=dim, hidden_dim=2048, output_dim=dim)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        self.memory_bank = NNMemoryBankModule(size=self.K, topk=self.topk).cuda()

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def _build_mlp(self, num_layers, input_dim, hidden_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else hidden_dim
            dim2 = output_dim if l == num_layers - 1 else hidden_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _generation_mask(self, batch_size):
        '''
        generation_mask(3) # topk = 2
        Out[5]:
        tensor([[   0.,    0., -999., -999., -999., -999.],
                [-999., -999.,    0.,    0., -999., -999.],
                [-999., -999., -999., -999.,    0.,    0.]])
        '''
        mask_ = torch.eye(batch_size)
        mask_ = (mask_ - 1) * 999
        mask = mask_.repeat(self.topk, 1).reshape(self.topk, batch_size, -1).permute(2, 1, 0).reshape(batch_size, self.topk*batch_size)

        return mask

    def generation_mask_with_pos(self, querys, nearest_neighbors, batch_size):
        # ================normalized==================
        querys = nn.functional.normalize(querys, dim=1)
        nearest_neighbors = nn.functional.normalize(nearest_neighbors, dim=1)

        cross_attn = torch.einsum("nc,mc->nm", querys, nearest_neighbors)
        att_mask = self._generation_mask(batch_size=batch_size).cuda()

        cross_attn = F.softmax(cross_attn + att_mask, dim=1)
        # ============threshold=========================
        if self.threshold:
            threshold_mask = (cross_attn > (1.0 / self.topk)).type(torch.long)
            cross_attn = cross_attn * threshold_mask
        # ============insert one-hot encoding===========
        one_hot = torch.eye(batch_size).cuda()
        cross_attn_scores = torch.cat((one_hot.reshape(batch_size, -1, batch_size).unsqueeze(dim=-1),
                            cross_attn.reshape(batch_size, -1, batch_size, self.topk)), dim=-1).reshape(batch_size, -1)

        return cross_attn_scores

    def contrastive_loss(self, im_q, im_k, labels, update=False):

        # compute query features
        z_q = self.projection_head(self.net(im_q))  # queries: NxC
        p_q = self.prediction_head(z_q)

        # z_k = self.projection_head(self.net(im_k))
        with torch.no_grad():  # no gradient to keys
            # shuffle
            im_k_, shuffle = batch_shuffle(im_k)
            z_k = self.projection_head_momentum(self.backbone_momentum(im_k_))  # keys: NxC
            # undo shuffle
            z_k = batch_unshuffle(z_k, shuffle)

        batch_size, _ = z_q.shape

        # Nearest Neighbour
        z_k_nn, purity, _, z_k_keep_nn = self.memory_bank(querys=z_k, keys=z_k, update=update, labels=labels)
        # ================normalized==================
        p_q = nn.functional.normalize(p_q, dim=1)
        z_k_keep_nn = nn.functional.normalize(z_k_keep_nn, dim=1)
        # calculate similiarities, has shape (n, n*(1+topk))
        logits = torch.einsum("nc,mc->nm", p_q, z_k_keep_nn) / self.tem
        # create labels
        cross_attn_scores = self.generation_mask_with_pos(querys=z_q, nearest_neighbors=z_k_nn, batch_size=batch_size).cuda()

        loss = - torch.sum(cross_attn_scores * F.log_softmax(logits, dim=1), dim=1).mean().cuda()

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