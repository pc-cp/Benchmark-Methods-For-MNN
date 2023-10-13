""" Nearest Neighbour Memory Bank Module """

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

import torch

from util.MemoryBankModule import MemoryBankModule

class NNMemoryBankModule(MemoryBankModule):
    """Nearest Neighbour Memory Bank implementation
        This code was taken and adapted from here:
        https://github.com/lightly-ai/lightly/blob/master/lightly/models/modules/nn_memory_bank.py
        https://github.com/ChongjianGE/SNCLR/blob/main/snclr/nn_memory_norm_bank_multi_keep.py

    This class implements a nearest neighbour memory bank as described in the
    NNCLR paper[0]. During the forward pass we return the nearest neighbour
    from the memory bank. But we improve the class so that it returns the
    neighbors of the topk of the query sample instead of 1.

    [0] NNCLR, 2021, https://arxiv.org/abs/2104.14548

    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0, memory bank is not used.
        topk:
            Number of neighbors of the query sample

    Examples:
        >>> memory_bank = NNMemoryBankModule(size=2 ** 16, topk=5)
        >>> z_k_nn, purity, _, _ = self.memory_bank(querys=z_k, keys=z_k, update=update, labels=labels)
    """

    def __init__(self, size: int = 2**16, topk: int = 1):
        super(NNMemoryBankModule, self).__init__(size)
        self.size = size
        self.topk = topk
    # Using query to find top-K neighbors in a bank composed of keys
    def forward(self, querys: torch.Tensor, keys: torch.Tensor, update: bool = False, labels = None):
        """Returns top-K neighbors of query that come from a bank composed of keys

        Args:
            querys: Tensors that need to find their nearest neighbors
            keys:   Sharing labels with query and may be in the queue if update is True
                Usually querys and keys are the same. Here querys is used to find its top-k neighbors in the memory
                bank and add keys to the memory bank after the find is done.

            labels: The true label shared by the current batch of querys and keys, which is used to calculate purity
                    and may also be queued if update is True
            update: If `True` updated the memory bank by adding keys and labels to it
        """

        # If update is True, enqueue the keys and labels, otherwise we just return the keys, memory bank and labels bank
        keys, bank, bank_labels = super(NNMemoryBankModule, self).forward(output=keys, labels=labels, update=update)

        bank = bank.to(keys.device).t() # [feature_dim, size] -> [size, feature_dim]
        bank_labels = bank_labels.to(keys.device)

        query_normed = torch.nn.functional.normalize(querys, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)
        similarity_matrix = torch.einsum("nd,md->nm", query_normed, bank_normed)

        _, index_nearest_neighbours = similarity_matrix.topk(self.topk, dim=1, largest=True)

        batch_size, feature_dim = querys.shape
        current_batch_labels = labels.unsqueeze(1).expand(batch_size, self.topk)
        labels_queue = bank_labels.clone().detach()
        labels_queue = labels_queue.unsqueeze(0).expand((batch_size, self.size))
        labels_queue = torch.gather(labels_queue, dim=1, index=index_nearest_neighbours)
        matches = (labels_queue == current_batch_labels).float()
        purity = (matches.sum(dim=1) / self.topk).mean()

        # nearest_neighbours.shape: [batch_size*topk, feature_dim]
        nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours.reshape(-1))
        # nearest_neighbours.shape: [batch_size*(topk+1), feature_dim]
        add_output_to_nearest_neighbours = torch.cat((querys.unsqueeze(dim=1),
                                                      nearest_neighbours.reshape(batch_size, self.topk, feature_dim)),
                                                     dim=1).reshape(-1, feature_dim)

        return nearest_neighbours, purity, bank, add_output_to_nearest_neighbours

    '''
    #   相似性计算 【batch_size, queue_size】
    similarity_matrix[i,j]:batch处理中第i个图像与queue中第j个图像的相似度
    similarity_matrix = torch.rand(3, 9)
    Out[8]: 
tensor([[0.6533, 0.9244, 0.8312, 0.7199, 0.8055, 0.9284, 0.1710, 0.6206, 0.3485],
        [0.6878, 0.2597, 0.6803, 0.7734, 0.6044, 0.0977, 0.2780, 0.7126, 0.7736],
        [0.3891, 0.4165, 0.1954, 0.7272, 0.3553, 0.7526, 0.6881, 0.9822, 0.9788]])
    找出similarity_matrix每行中前topk个最大的，其中index_nearest_neighbours指的是相似度矩阵的对应下标
    _, index_nearest_neighbours = similarity_matrix.topk(4, dim=1, largest=True)
    index_nearest_neighbours
Out[10]: 
tensor([[5, 1, 2, 4],
        [8, 3, 7, 0],
        [7, 8, 5, 3]])
    batch_size, feature_dim = torch.tensor(3), torch.tensor(2)
    # 当前batch_size对应的标签
    labels = torch.tensor([0, 2, 1])
    current_batch_labels = labels.unsqueeze(1).expand(batch_size, 4)
    Out[14]: 
tensor([[0, 0, 0, 0],
        [2, 2, 2, 2],
        [1, 1, 1, 1]])
    # 队列中元素对应的标签
    labels_queue = torch.tensor([0, 2, 4, 1, 6, 2, 3, 8, 7])
    labels_queue = labels_queue.unsqueeze(0).expand((batch_size, 9))
    labels_queue
Out[24]: 
tensor([[0, 2, 4, 1, 6, 2, 3, 8, 7],
        [0, 2, 4, 1, 6, 2, 3, 8, 7],
        [0, 2, 4, 1, 6, 2, 3, 8, 7]])
    labels_queue = torch.gather(labels_queue, dim=1, index=index_nearest_neighbours)
Out[25]: 
tensor([[2, 2, 4, 6],
        [7, 1, 8, 0],
        [8, 7, 2, 1]])
        
    matches = (labels_queue == current_batch_labels).float()
matches
Out[31]: 
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 1.]])

bank = torch.rand(9, 2)
bank
Out[33]: 
tensor([[0.7875, 0.3157],
        [0.7495, 0.8160],
        [0.2900, 0.9952],
        [0.6331, 0.8462],
        [0.3460, 0.2054],
        [0.6855, 0.5992],
        [0.1961, 0.2234],
        [0.3103, 0.3423],
        [0.5443, 0.2770]])
nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours.reshape(-1))
nearest_neighbours
Out[35]: 
tensor([[0.6855, 0.5992],
        [0.7495, 0.8160],
        [0.2900, 0.9952],
        [0.3460, 0.2054],
        [0.5443, 0.2770],
        [0.6331, 0.8462],
        [0.3103, 0.3423],
        [0.7875, 0.3157],
        [0.3103, 0.3423],
        [0.5443, 0.2770],
        [0.6855, 0.5992],
        [0.6331, 0.8462]])
    '''
