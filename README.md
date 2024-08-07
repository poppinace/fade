# FADE: Fusing the Assets of Decoder and Encoder for Task-Agnostic Upsampling

<p align="center"><img src="carafe.gif" width="400" title="CARAFE"/><img src="fade.gif" width="400" title="FADE"/></p>
<p align="center">CARAFE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FADE</p>

This repository includes the official implementation of FADE, an upsampling operator, presented in our paper:

**[FADE: Fusing the Assets of Decoder and Encoder for Task-Agnostic Upsampling](http://arxiv.org/abs/2207.10392)**

and the extended version

**[FADE: A Task-Agnostic Upsampling Operator for Encoder–Decoder Architectures](https://link.springer.com/article/10.1007/s11263-024-02191-8)**

Proc. European Conference on Computer Vision (ECCV) / International Journal of Computer Vision

[Hao Lu](https://sites.google.com/site/poppinace/), Wenze Liu, Hongtao Fu, Zhiguo Cao

Huazhong University of Science and Technology, China

## Highlights
- **Simple and effective:** As an upsampling operator, FADE boosts great improvements despite its tiny body;
- **Task-agnostic:** Compared with other upsamplers, FADE performs well on both region-sensitive and detail sensitive dense prediction tasks;
- **Plug and play:** FADE can be easily incorporated into most dense prediction models, particularly encoder-decoder architectures.
<p align="center"><img src="visualization.jpg" width="800" title="visualization"/></p>

## Installation
Our codes are tested on Python 3.8.8 and PyTorch 1.9.0. [mmcv](https://github.com/open-mmlab/mmcv) is additionally required for the feature assembly function by [CARAFE](https://github.com/myownskyW7/CARAFE).

## Start
Our experiments are based on [A2U matting](https://github.com/dongdong93/a2u_matting) and [SegFormer](https://github.com/NVlabs/SegFormer). Please follow their installation instructions to prepare the models. In the folders `a2u_matting` and `segformer` we provide the modified model and the config files for FADE and FADE-Lite.

Here are results of image matting and semantic segmentation:

| Image Matting | #Param. | GFLOPs | SAD | MSE | Grad | Conn | Log | 
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| Bilinear | 8.05M | 8.61 | 37.31 | 0.0103 | 21.38 | 35.39 | -- |
| CARAFE | +0.26M | +6.00 | 41.01 | 0.0118 | 21.39 | 39.01 | -- |
| IndexNet | +12.26M | +31.70 | 34.28 | 0.0081 | 15.94 | 31.91 | -- |
| A2U | +38K | +0.66 | 32.15 | 0.0082 | 16.39 | 29.25 | -- |
| FADE | +0.12M | +8.85 | 31.10 | 0.0073 | 14.52 | 28.11 | [link](https://github.com/poppinace/fade/blob/main/a2u_matting/matting_fade.txt) |
| FADE-Lite | +27K | +1.46 | 31.36 | 0.0075 | 14.83 | 28.21 | [link](https://github.com/poppinace/fade/blob/main/a2u_matting/matting_fade_lite.txt) |

| Semantic Segmentation | #Param. | GFLOPs | mIoU | bIoU | Log |
| :--: | :--: | :--: | :--: | :--: | :--: |
| Bilinear | 13.7M | 15.91 | 41.68 | 27.80 | [link](https://github.com/poppinace/fade/blob/main/segformer/segformer.b1_bilinear.512x512.ade.160k.log) |
| CARAFE | +0.44M | +1.45 | 42.82 | 29.84 | [link](https://github.com/poppinace/fade/blob/main/segformer/segformer.b1_carafe.512x512.ade.160k.log) |
| IndexNet | +12.60M | +30.65 | 41.50 | 28.27 | [link](https://github.com/poppinace/fade/blob/main/segformer/segformer.b1_indexnet.512x512.ade.160k.log) |
| A2U | +0.12M | +0.41 | 41.45 | 27.31 | [link](https://github.com/poppinace/fade/blob/main/segformer/segformer.b1_a2u.512x512.ade.160k.log) |
| FADE | +0.29M | +2.65 | 44.41 | 32.65 | [link](https://github.com/poppinace/fade/blob/main/segformer/segformer.b1_fade.512x512.ade.160k.log) |
| FADE-Lite | +80K | +0.89 | 43.49 | 31.55 | [link](https://github.com/poppinace/fade/blob/main/segformer/segformer.b1_fade_lite.512x512.ade.160k.log) |


## Citation
If you find this work or code useful for your research, please cite:
```
@inproceedings{lu2022fade,
  title={FADE: Fusing the Assets of Decoder and Encoder for Task-Agnostic Upsampling},
  author={Lu, Hao and Liu, Wenze and Fu, Hongtao and Cao, Zhiguo},
  booktitle={Proc. European Conference on Computer Vision (ECCV)},
  year={2022}
}

@article{lu2024fade,
  title={FADE: A Task-Agnostic Upsampling Operator for Encoder–Decoder Architectures},
  author={Lu, Hao and Liu, Wenze and Fu, Hongtao and Cao, Zhiguo},
  journal={International Journal of Computer Vision},
  year={2024}
}
```

## Permission
This code is for academic purposes only. Contact: Hao Lu (hlu@hust.edu.cn)
