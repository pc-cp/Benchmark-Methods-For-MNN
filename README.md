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

## Reproduced results
| Method        | CIFAR-10(200-NN) | CIFAR-10(Linear-evaluation) | CIFAR-100(200-NN) | CIFAR-100(Linear-evaluation) | Tiny-ImageNet(200-NN) | Tiny-ImageNet(Linear-evaluation) | 
|---------------|------------------|-----------------------------|-------------------|------------------------------|-----------------------|----------------------------------|
| MoCoV2        | 87.82            | 89.56                       | 57.29             | 62.47                        | 37.77                 | 46.38                            |
| SimCLR        | 85.30            | 87.72                       | 56.50             | 62.52                        | 37.16                 | 45.71                            |
| BYOL          | 87.54            | 89.54                       | 57.24             | 63.16                        | 37.65                 | 45.29                            |
| ReSSL         | 88.89            | 89.97                       | 57.48             | 63.82                        | 37.14                 | 46.38                            |
| NNCLR(topk=1) | 85.19-0.81       | 87.72                       | 50.54-0.45        | 59.62                        | 30.93-0.30            | 41.52                            |
| NNCLR(topk=5) | 86.98-0.81       | 88.83                       | 53.40-0.36        | 62.54                        | 34.99-0.19            | 44.42                            |
| MSF(topk=5)   | 88.24-0.83       | 89.94                       | 52.32-0.35        | 59.94                        | 35.29-0.20            | 42.68                            |
| SNCLR(topk=5) | 87.36-0.82       | 88.86                       | 58.65-0.48        | 65.19                        | 41.92-0.33            | 50.15                            |

## Algorithms reproduced soon after
| Source                                         | Reference                                                                      | Algorithm |
|------------------------------------------------|--------------------------------------------------------------------------------|-----------|
| [WACV'23](https://arxiv.org/abs/2111.14585)    | [code](https://github.com/CEA-LIST/SCE)                                        | SCE       |