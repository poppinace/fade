# FADE: Fusing the Assets of Decoder and Encoder for Task-Agnostic Upsampling

<p align="center"><img src="carafe.gif" width="400" title="CARAFE"/><img src="fade.gif" width="400" title="FADE"/></p>
<p align="center">CARAFE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FADE</p>

This repository includes the official implementation of FADE, an upsampling operator, presented in our paper:

**[FADE: Fusing the Assets of Decoder and Encoder for Task-Agnostic Upsampling](https://arxiv.org/abs/)**

Proc. ECCV European Conference on Computer Vision

[Hao Lu](https://sites.google.com/site/poppinace/)<sup>1</sup>, Wenze Liu<sup>1</sup>, Hongtao Fu<sup>1</sup>, Zhiguo Cao<sup>1</sup>

<sup>1</sup>Huazhong University of Science and Technology, China

## Highlights
- **Simple and effective:** As an upsampling operator, FADE boosts great improvements despite its tiny body;
- **Task-agnostic:** Compared with other upsamplers, FADE performs well on both region-sensitive and detail sensitive dense prediction tasks.
- **Plug and play:** FADE is easily equipped on most of dense prediction models.

## Installation
Our codes are tested on Python 3.8.8 and PyTorch 1.9.0. mmcv(https://github.com/open-mmlab/mmcv) is additionally required for the feature assembly function by CARAFE(https://github.com/myownskyW7/CARAFE).

## Start
Our experiments are based on A2U matting(https://github.com/dongdong93/a2u_matting) and SegFormer(https://github.com/NVlabs/SegFormer). Please follow their installation instruction to prepare the models. In the folders a2u_matting and segformer we provide the modified model and the config files for FADE and FADE-Lite.

## Citation
If you find this work or code useful for your research, please cite:
```
@inproceedings{
}
