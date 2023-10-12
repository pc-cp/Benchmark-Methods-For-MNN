# Mini-SSL

<picture>
  <source
    srcset="https://github-readme-stats.vercel.app/api?username=pc-cp&show_icons=true&theme=dark"
    media="(prefers-color-scheme: dark)"
  />
  <source
    srcset="https://github-readme-stats.vercel.app/api?username=pc-cp&show_icons=true"
    media="(prefers-color-scheme: light), (prefers-color-scheme: no-preference)"
  />
  <img src="https://github-readme-stats.vercel.app/api?username=pc-cp&show_icons=true" />
</picture>

It mainly consists of fast reproduction of classical algorithms in the field of self-supervised learning with small models and lightweight datasets.


## Reproduced algorithms
| Source                                      | Reference                                         | Algorithm | Experimental records       | Checkpoint                                                                                           |
|---------------------------------------------|---------------------------------------------------|-----------|----------------------------|------------------------------------------------------------------------------------------------------|
| [CVPR'20](https://arxiv.org/abs/1911.05722) | [code](https://github.com/facebookresearch/moco)  | MoCoV2    | cifar10_1/cifar100_1/tin_1 | [link-moco](https://drive.google.com/drive/folders/17tcUy1nWO4_KwTVWV-bwNo0gw59CHe_X?usp=share_link) |
| [ICML'20](https://arxiv.org/abs/2002.05709) | [code](https://github.com/google-research/simclr) | SimCLR    | cifar10_2/cifar100_2/tin_2 | [link-simclr](https://drive.google.com/drive/folders/1fWiCbx30UDUmmnNxXArf32LCem1PCmT-?usp=sharing)  |

## Reproduced results
| Method | CIFAR-10(200-NN) | CIFAR-10(Linear-evaluation) | CIFAR-100(200-NN) | CIFAR-100(Linear-evaluation) | Tiny-ImageNet(200-NN) | Tiny-ImageNet(Linear-evaluation) | 
|--------|------------------|-----------------------------|-------------------|------------------------------|-----------------------|----------------------------------|
| MoCoV2 | 87.82            | 89.56                       | 57.29             | 62.47                        | 37.77                 | 46.38                            |
| SimCLR | 85.30            | 87.72                       | 56.50             | 62.52                        | 37.16                 | 45.71                            |

## Algorithms reproduced soon after
| Source                                                                                                                                                                      | Reference                                                                      | Algorithm |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|-----------|
| [NeurIPS'20](https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf)                                                                             | [code](https://github.com/google-deepmind/deepmind-research/tree/master/byol ) | BYOL      |
| [ICCV'21](https://arxiv.org/abs/2104.14548)                                                                                                                                 | [code](https://docs.lightly.ai/self-supervised-learning/examples/nnclr.html)   | NNCLR     |
| [ICCV'21](https://arxiv.org/pdf/2105.07269.pdf)                                                                                                                             | [code](https://github.com/UMBCvision/MSF)                                      | MSF       |
| [NeurIPS'21](https://proceedings.neurips.cc/paper/2021/file/14c4f36143b4b09cbc320d7c95a50ee7-Paper.pdf)                                                                     | [code](https://github.com/mingkai-zheng/ReSSL)                                 | ReSSL     |
| [ICLR'23](https://arxiv.org/abs/2303.17142)                                                                                                                                 | [code](https://github.com/ChongjianGE/SNCLR)                                   | SNCLR     |
| [WACV'23](https://openaccess.thecvf.com/content/WACV2023/papers/Denize_Similarity_Contrastive_Estimation_for_Self-Supervised_Soft_Contrastive_Learning_WACV_2023_paper.pdf) | [code](https://github.com/CEA-LIST/SCE)                                        | SCE       |