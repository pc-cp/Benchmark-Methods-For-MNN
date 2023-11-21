# Mini-SSL
## Updata(21,Nov, 2023)
We fixed bugs in some algorithms (**MoCoV2, NNCLR, MSF, SNCLR, SCE, CMSF**) which as you can see were hidden deeper but had little impact on performance. The bug is that the samples in the batch are updated in the queue during the symmetric computation of the loss function
The fix for this bug is to change clip 1 to clip 2:

clip 1
```python
loss_12 = self.contrastive_loss(im1, im2, labels=labels, update=True)
loss = loss_12
if self.symmetric:  # symmetric loss
    loss_21 = self.contrastive_loss(im2, im1,  labels=labels, update=False)
    loss = (loss_12 + loss_21)*1.0/2
```
clip 2
```python
if self.symmetric:  # symmetric loss
    loss_21 = self.contrastive_loss(im2, im1,  labels=labels, update=False)

loss_12 = self.contrastive_loss(im1, im2, labels=labels, update=True)
loss = loss_12

# compute loss
if self.symmetric:  # symmetric loss
    # loss_21 = self.contrastive_loss(im2, im1,  labels=labels, update=False)
    loss = (loss_12 + loss_21)*1.0/2
```

<picture>
  <source
    srcset="https://github-readme-stats.vercel.app/api?username=pc-cp&show_icons=true&icon_color=CE1D2D&text_color=718096&hide_title=true&theme=dark"
    media="(prefers-color-scheme: dark)"
  />
  <source
    srcset="https://github-readme-stats.vercel.app/api?username=pc-cp&show_icons=true&icon_color=CE1D2D&text_color=718096&hide_title=true"
    media="(prefers-color-scheme: light), (prefers-color-scheme: no-preference)"
  />
  <img align="right" src="https://github-readme-stats.vercel.app/api?username=pc-cp&show_icons=true&icon_color=CE1D2D&text_color=718096&hide_title=true" />
</picture>

It mainly consists of fast reproduction of classical algorithms in the field of self-supervised learning with small models and lightweight datasets. If you find this repo useful, welcome 🌟🌟🌟.


## Reproduced algorithms
| Source                                         | Reference                                                                     | Algorithm | Experimental records                     | Checkpoint                                                                                            |
|------------------------------------------------|-------------------------------------------------------------------------------|-----------|------------------------------------------|-------------------------------------------------------------------------------------------------------|
| [CVPR'20](https://arxiv.org/abs/1911.05722)    | [code](https://github.com/facebookresearch/moco)                              | MoCoV2    | cifar10_1/cifar100_1/tin_1               | [link-moco](https://drive.google.com/drive/folders/17tcUy1nWO4_KwTVWV-bwNo0gw59CHe_X?usp=share_link)  |
| [ICML'20](https://arxiv.org/abs/2002.05709)    | [code](https://github.com/google-research/simclr)                             | SimCLR    | cifar10_2/cifar100_2/tin_2               | [link-simclr](https://drive.google.com/drive/folders/1fWiCbx30UDUmmnNxXArf32LCem1PCmT-?usp=sharing)   |
| [NeurIPS'20](https://arxiv.org/abs/2006.07733) | [code](https://github.com/google-deepmind/deepmind-research/tree/master/byol) | BYOL      | cifar10_3/cifar100_3/tin_3               | [link-byol](https://drive.google.com/drive/folders/11Rq_hBn3Ce3wcLOHxjwr3ZHKrXMaX7tX?usp=sharing)     |
| [NeurIPS'21](https://arxiv.org/abs/2107.09282) | [code](https://github.com/mingkai-zheng/ReSSL)                                | ReSSL     | cifar10_4/cifar100_4/tin_4               | [link-ressl](https://drive.google.com/drive/folders/1v_fkA1V05G79bHwC_EmUrChbTfqKGmqx?usp=sharing)    |
| [ICCV'21](https://arxiv.org/abs/2104.14548)    | [code](https://docs.lightly.ai/self-supervised-learning/examples/nnclr.html)  | NNCLR     | cifar10_5/cifar100_5/tin_5, topk=1       | [link-nnclr](https://drive.google.com/drive/folders/1WuUige97xwyQ6fLEHX86ioS_3qfmi8ZX?usp=sharing)    |
|                                                |                                                                               |           | cifar10_5a/cifar100_5a/tin_5a, topk=5    | [link-nnclr(a)](https://drive.google.com/drive/folders/1_Kz2EMec7pPdnSfYp1Pp9T9gQeus6oaA?usp=sharing) |
| [ICCV'21](https://arxiv.org/abs/2105.07269)    | [code](https://github.com/UMBCvision/MSF)                                     | MSF       | cifar10_6/cifar100_6/tin_6, topk=5, weak | [link-msf](https://drive.google.com/drive/folders/16FfJPM59G5fr2i43Uua9XXtaE8q72t1K?usp=sharing)      |
| [ICLR'23](https://arxiv.org/abs/2303.17142)    | [code](https://github.com/ChongjianGE/SNCLR)                                  | SNCLR     | cifar10_7/cifar100_7/tin_7, topk=5       | [link-snclr](https://drive.google.com/drive/folders/1XH_nh1ToQatNdB8MkF_db_MInLsCRQGh?usp=sharing)    |
| [WACV'23](https://arxiv.org/abs/2111.14585)    | [code](https://github.com/CEA-LIST/SCE)                                       | SCE       | cifar10_8/cifar100_8/tin_8               | [link-sce](https://drive.google.com/drive/folders/19akqdMiDw-2MgOUzBfeb5OxQmJzLxqPW?usp=sharing)      |
| [ECCV'22](https://arxiv.org/abs/2112.04607)    | [code](https://github.com/UCDvision/CMSF)                                     | CMSF      | cifar10_9/cifar100_9/tin_9               | [link-cmsf](https://drive.google.com/drive/folders/15o1k8YTxTdScUYYp_hRyXKubToqeCzHJ?usp=sharing)     |
| [AAAI'22](https://arxiv.org/abs/2003.05438)    | [code](https://github.com/szq0214/Un-Mix)                                     | UnMix     | cifar10_10/cifar100_10/tin_10            | [link-unmix](https://drive.google.com/drive/folders/1wCHsGdnd58Np2JmJTHkzcHSSeJEMEt-1?usp=sharing)    |

## Reproduced results
It is worth noting that the results reported by **MoCoV2, SimCLR, and BYOL** may be **somewhat high**, and you can refer to the results reported in this paper [ReSSL](https://arxiv.org/abs/2107.09282) for comparison.

| Method        | CIFAR-10(200-NN) | CIFAR-10(Linear-evaluation) | CIFAR-100(200-NN) | CIFAR-100(Linear-evaluation) | Tiny-ImageNet(200-NN) | Tiny-ImageNet(Linear-evaluation) | 
|---------------|------------------|-----------------------------|-------------------|------------------------------|-----------------------|----------------------------------|
| MoCoV2        | 88.05            | 89.70                       | 57.64             | 62.37                        | 37.29                 | 46.08                            |
| SimCLR        | 85.30            | 87.72                       | 56.50             | 62.52                        | 37.16                 | 45.71                            |
| BYOL          | 87.54            | 89.54                       | 57.24             | 63.16                        | 37.65                 | 45.29                            |
| ReSSL         | 88.89            | 89.97                       | 57.48             | 63.82                        | 37.14                 | 46.38                            |
| NNCLR(topk=1) | 85.47            | 88.06                       | 50.99             | 60.53                        | 31.62                 | 41.70                            |
| NNCLR(topk=5) | 86.78            | 88.80                       | 53.21             | 61.25                        | 34.44                 | 44.38                            |
| MSF(topk=5)   | 87.86            | 89.60                       | 51.29             | 59.84                        | 34.79                 | 42.52                            |
| SNCLR(topk=5) | 87.64            | 89.64                       | 58.57             | 65.45                        | 41.42                 | 49.62                            |
| SCE           | 89.15            | 90.62                       | 60.17             | 64.93                        | 40.66                 | 48.63                            |
| CMSF          | 88.95            | 90.63                       | 54.20             | 61.24                        | 35.97                 | 44.00                            |
| UnMix         | 87.99            | 90.37                       | 59.11             | 65.30                        | 38.65                 | 47.29                            |

## Algorithms reproduced soon after
| Source                                       | Reference                                   | Algorithm |
|----------------------------------------------|---------------------------------------------|-----------|
