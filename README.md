# Mini-SSL

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

It mainly consists of fast reproduction of classical algorithms in the field of self-supervised learning with small models and lightweight datasets. If you find this repo useful, welcome 🌟


## Reproduced algorithms
| Source                                          | Reference                                                                     | Algorithm | Experimental records       | Checkpoint                                                                                            |
|-------------------------------------------------|-------------------------------------------------------------------------------|-----------|----------------------------|-------------------------------------------------------------------------------------------------------|
| [CVPR'20](https://arxiv.org/abs/1911.05722)     | [code](https://github.com/facebookresearch/moco)                              | MoCoV2    | cifar10_1/cifar100_1/tin_1 | [link-moco](https://drive.google.com/drive/folders/17tcUy1nWO4_KwTVWV-bwNo0gw59CHe_X?usp=share_link)  |
| [ICML'20](https://arxiv.org/abs/2002.05709)     | [code](https://github.com/google-research/simclr)                             | SimCLR    | cifar10_2/cifar100_2/tin_2 | [link-simclr](https://drive.google.com/drive/folders/1fWiCbx30UDUmmnNxXArf32LCem1PCmT-?usp=sharing)   |
| [NeurIPS'20](https://arxiv.org/abs/2006.07733)  | [code](https://github.com/google-deepmind/deepmind-research/tree/master/byol) | BYOL      | cifar10_3/cifar100_3/tin_3 | [link-byol](https://drive.google.com/drive/folders/11Rq_hBn3Ce3wcLOHxjwr3ZHKrXMaX7tX?usp=sharing)     |

## Reproduced results
| Method | CIFAR-10(200-NN) | CIFAR-10(Linear-evaluation) | CIFAR-100(200-NN) | CIFAR-100(Linear-evaluation) | Tiny-ImageNet(200-NN) | Tiny-ImageNet(Linear-evaluation) | 
|--------|------------------|-----------------------------|-------------------|------------------------------|-----------------------|----------------------------------|
| MoCoV2 | 87.82            | 89.56                       | 57.29             | 62.47                        | 37.77                 | 46.38                            |
| SimCLR | 85.30            | 87.72                       | 56.50             | 62.52                        | 37.16                 | 45.71                            |
| BYOL   | 87.54            | 89.54                       | 57.24             | 63.16                        | 37.65                 | 45.29                            |

## Algorithms reproduced soon after
| Source                                         | Reference                                                                      | Algorithm |
|------------------------------------------------|--------------------------------------------------------------------------------|-----------|
| [ICCV'21](https://arxiv.org/abs/2104.14548)    | [code](https://docs.lightly.ai/self-supervised-learning/examples/nnclr.html)   | NNCLR     |
| [ICCV'21](https://arxiv.org/abs/2105.07269)    | [code](https://github.com/UMBCvision/MSF)                                      | MSF       |
| [NeurIPS'21](https://arxiv.org/abs/2107.09282) | [code](https://github.com/mingkai-zheng/ReSSL)                                 | ReSSL     |
| [ICLR'23](https://arxiv.org/abs/2303.17142)    | [code](https://github.com/ChongjianGE/SNCLR)                                   | SNCLR     |
| [WACV'23](https://arxiv.org/abs/2111.14585)    | [code](https://github.com/CEA-LIST/SCE)                                        | SCE       |