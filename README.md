<div align="center">

<img align="left" width="80" height="80" src="https://github.com/gy65896/OneRestore/blob/main/img_file/logo_onerestore.png" alt="">

 # <p align=center> [ECCV 2024] OneRestore: A Universal Restoration Framework for Composite Degradation</p>
 
[![ArXiv](https://img.shields.io/badge/OneRestore-ArXiv-red.svg)](https://arxiv.org/abs/2407.04621)
[![Paper](https://img.shields.io/badge/OneRestore-Paper-yellow.svg)](https://link.springer.com/chapter/10.1007/978-3-031-72655-2_15)
[![Web](https://img.shields.io/badge/OneRestore-Web-blue.svg)](https://gy65896.github.io/projects/ECCV2024_OneRestore/index.html)
[![Poster](https://img.shields.io/badge/OneRestore-Poster-green.svg)](https://github.com/gy65896/OneRestore/blob/main/img_file/OneRestore_poster.png)
[![Video](https://img.shields.io/badge/OneRestore-Video-orange.svg)](https://www.youtube.com/watch?v=AFr5tZdPlZ4)

<!--[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgy65896%2FOneRestore&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)-->
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/gy65896/OneRestore)
[![Closed Issues](https://img.shields.io/github/issues-closed/gy65896/OneRestore)](https://github.com/gy65896/OneRestore/issues?q=is%3Aissue+is%3Aclosed)
[![Open Issues](https://img.shields.io/github/issues/gy65896/OneRestore)](https://github.com/gy65896/OneRestore/issues)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/onerestore-a-universal-restoration-framework/image-restoration-on-cdd-11)](https://paperswithcode.com/sota/image-restoration-on-cdd-11?p=onerestore-a-universal-restoration-framework)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/onerestore-a-universal-restoration-framework/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=onerestore-a-universal-restoration-framework)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/onerestore-a-universal-restoration-framework/image-dehazing-on-sots-outdoor)](https://paperswithcode.com/sota/image-dehazing-on-sots-outdoor?p=onerestore-a-universal-restoration-framework)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/onerestore-a-universal-restoration-framework/rain-removal-on-did-mdn)](https://paperswithcode.com/sota/rain-removal-on-did-mdn?p=onerestore-a-universal-restoration-framework)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/onerestore-a-universal-restoration-framework/snow-removal-on-snow100k)](https://paperswithcode.com/sota/snow-removal-on-snow100k?p=onerestore-a-universal-restoration-framework)

</div>
<div align=center>
<img src="https://github.com/gy65896/OneRestore/blob/main/img_file/abstract.jpg" width="720">
</div>

---
>**OneRestore: A Universal Restoration Framework for Composite Degradation**<br>  [Yu Guo](https://scholar.google.com/citations?user=klYz-acAAAAJ&hl=zh-CN)<sup>† </sup>, [Yuan Gao](https://scholar.google.com.hk/citations?user=4JpRnU4AAAAJ&hl=zh-CN)<sup>† </sup>, [Yuxu Lu](https://scholar.google.com.hk/citations?user=XXge2_0AAAAJ&hl=zh-CN), [Huilin Zhu](https://scholar.google.com.hk/citations?hl=zh-CN&user=fluPrxcAAAAJ), [Ryan Wen Liu](http://mipc.whut.edu.cn/index.html)<sup>* </sup>, [Shengfeng He](http://www.shengfenghe.com/)<sup>* </sup> <br>
(† Co-first Author, * Corresponding Author)<br>
>European Conference on Computer Vision

> **Abstract:** *In real-world scenarios, image impairments often manifest as composite degradations, presenting a complex interplay of elements such as low light, haze, rain, and snow. Despite this reality, existing restoration methods typically target isolated degradation types, thereby falling short in environments where multiple degrading factors coexist. To bridge this gap, our study proposes a versatile imaging model that consolidates four physical corruption paradigms to accurately represent complex, composite degradation scenarios. In this context, we propose OneRestore, a novel transformer-based framework designed for adaptive, controllable scene restoration. The proposed framework leverages a unique cross-attention mechanism, merging degraded scene descriptors with image features, allowing for nuanced restoration. Our model allows versatile input scene descriptors, ranging from manual text embeddings to automatic extractions based on visual attributes. Our methodology is further enhanced through a composite degradation restoration loss, using extra degraded images as negative samples to fortify model constraints. Comparative results on synthetic and real-world datasets demonstrate OneRestore as a superior solution, significantly advancing the state-of-the-art in addressing complex, composite degradations.*
---

## News 🚀
* **2024.09.07**: [Hugging Face Demo](https://huggingface.co/spaces/gy65896/OneRestore) is released.
* **2024.09.05**: Video and poster are released.
* **2024.09.04**: Code for data synthesis is released.
* **2024.07.27**: Code for multiple GPUs training is released.
* **2024.07.20**: [New Website](https://gy65896.github.io/projects/ECCV2024_OneRestore) has been created.
* **2024.07.10**: [Paper](https://arxiv.org/abs/2407.04621) is released on ArXiv.
* **2024.07.07**: Code and Dataset are released.
* **2024.07.02**: OneRestore is accepted by [ECCV2024](https://eccv.ecva.net/).

## Network Architecture

</div>
<div align=center>
<img src="https://github.com/gy65896/OneRestore/blob/main/img_file/pipeline.jpg" width="1080">
</div>

## Quick Start

### Install

- python 3.7
- cuda 11.7

```
# git clone this repository
git clone https://github.com/gy65896/OneRestore.git
cd OneRestore

# create new anaconda env
conda create -n onerestore python=3.7
conda activate onerestore 

# download ckpts
put embedder_model.tar and onerestore_cdd-11.tar in ckpts folder

# install pytorch (Take cuda 11.7 as an example to install torch 1.13)
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# install other packages
pip install -r requirements.txt
pip install gensim
```

### Pretrained Models

Please download our pre-trained models and put them in  `./ckpts`.

| Model | OneDrive | Hugging Face| Description
| :--- | :--- | :--- | :----------
|embedder_model.tar | [model](https://1drv.ms/u/s!As3rCDROnrbLgqpnhSQFIoD9msXWOA?e=aUpHOT) | [model](https://huggingface.co/gy65896/OneRestore/tree/main)  | Text/Visual Embedder trained on our CDD-11.
|onerestore_cdd-11.tar | [model](https://1drv.ms/u/s!As3rCDROnrbLgqpmWkGBku6oj33efg?e=7yUGfN) | model | OneRestore trained on our CDD-11.
|onerestore_real.tar | [model](https://1drv.ms/u/s!As3rCDROnrbLgqpi-iJOyN6OSYqiaA?e=QFfMeL) | model | OneRestore trained on our CDD-11 for Real Scenes.
|onerestore_lol.tar | [model](https://1drv.ms/u/s!As3rCDROnrbLgqpkSoVB1j-wYHFpHg?e=0gR9pn) | model | OneRestore trained on LOL (low light enhancement benchmark).
|onerestore_reside_ots.tar | [model](https://1drv.ms/u/s!As3rCDROnrbLgqpjGh8KjfM_QIJzEw?e=zabGTw) | model | OneRestore trained on RESIDE-OTS (image dehazing benchmark).
|onerestore_rain1200.tar | [model](https://1drv.ms/u/s!As3rCDROnrbLgqplAFHv6B348jarGA?e=GuduMT) | model | OneRestore trained on Rain1200 (image deraining benchmark).
|onerestore_snow100k.tar | [model](https://1drv.ms/u/s!As3rCDROnrbLgqphsWWxLZN_7JFJDQ?e=pqezzo) | model | OneRestore trained on Snow100k-L (image desnowing benchmark).

### Inference

We provide two samples in `./image` for the quick inference:

```
python test.py --embedder-model-path ./ckpts/embedder_model.tar --restore-model-path ./ckpts/onerestore_cdd-11.tar --input ./image/ --output ./output/ --concat
```

You can also input the prompt to perform controllable restoration. For example:

```
python test.py --embedder-model-path ./ckpts/embedder_model.tar --restore-model-path ./ckpts/onerestore_cdd-11.tar --prompt low_haze --input ./image/ --output ./output/ --concat
```

## Training

### Prepare Dataset

We provide the download link of our Composite Degradation Dataset with 11 types of degradation **CDD-11** ([OneDrive](https://1drv.ms/f/s!As3rCDROnrbLgqpezG4sao-u9ddDhw?e=A0REHx) | [Hugging Face](https://huggingface.co/datasets/gy65896/CDD-11/tree/main)).

Preparing the train and test datasets as follows:

```
./data/
|--train
|  |--clear
|  |  |--000001.png
|  |  |--000002.png
|  |--low
|  |--haze
|  |--rain
|  |--snow
|  |--low_haze
|  |--low_rain
|  |--low_snow
|  |--haze_rain
|  |--haze_snow
|  |--low_haze_rain
|  |--low_haze_snow
|--test
```
### Train Model

**1. Train Text/Visual Embedder by**

```
python train_Embedder.py --train-dir ./data/CDD-11_train --test-dir ./data/CDD-11_test --check-dir ./ckpts --batch 256 --num-workers 0 --epoch 200 --lr 1e-4 --lr-decay 50
```

**2. Remove the optimizer weights in the Embedder model file by**

```
python remove_optim.py --type Embedder --input-file ./ckpts/embedder_model.tar --output-file ./ckpts/embedder_model.tar
```

**3. Generate the `dataset.h5` file for training OneRestore by**

```
python makedataset.py --train-path ./data/CDD-11_train --data-name dataset.h5 --patch-size 256 --stride 200
```

**4. Train OneRestore model by**

- **Single GPU**

```
python train_OneRestore_single-gpu.py --embedder-model-path ./ckpts/embedder_model.tar --save-model-path ./ckpts --train-input ./dataset.h5 --test-input ./data/CDD-11_test --output ./result/ --epoch 120 --bs 4 --lr 1e-4 --adjust-lr 30 --num-works 4
```

- **Multiple GPUs**

Assuming you train the OneRestore model using 4 GPUs (e.g., 0, 1, 2, and 3), you can use the following command. Note that the number of nproc_per_node should equal the number of GPUs.

```
CUDA_VISIBLE_DEVICES=0, 1, 2, 3 torchrun --nproc_per_node=4 train_OneRestore_multi-gpu.py --embedder-model-path ./ckpts/embedder_model.tar --save-model-path ./ckpts --train-input ./dataset.h5 --test-input ./data/CDD-11_test --output ./result/ --epoch 120 --bs 4 --lr 1e-4 --adjust-lr 30 --num-works 4
```

**5. Remove the optimizer weights in the OneRestore model file by**

```
python remove_optim.py --type OneRestore --input-file ./ckpts/onerestore_model.tar --output-file ./ckpts/onerestore_model.tar
```

### Customize your own composite degradation dataset

**1. Prepare raw data**

 - Collect your own clear images.
 - Generate the depth map based on [MegaDepth](https://github.com/zhengqili/MegaDepth).
 - Generate the light map based on [LIME](https://github.com/estija/LIME).
 - Generate the rain mask database based on [RainStreakGen](https://github.com/liruoteng/RainStreakGen?tab=readme-ov-file).
 - Download the snow mask database from [Snow100k](https://sites.google.com/view/yunfuliu/desnownet).

A generated example is as follows:

| Clear Image | Depth Map | Light Map | Rain Mask | Snow Mask
| :--- | :---| :---| :--- | :---
| <img src="https://github.com/gy65896/OneRestore/blob/main/img_file/clear_img.jpg" width="200"> | <img src="https://github.com/gy65896/OneRestore/blob/main/img_file/depth_map.jpg" width="200"> | <img src="https://github.com/gy65896/OneRestore/blob/main/img_file/light_map.jpg" width="200"> | <img src="https://github.com/gy65896/OneRestore/blob/main/img_file/rain_mask.jpg" width="200"> | <img src="https://github.com/gy65896/OneRestore/blob/main/img_file/snow_mask.png" width="200">

(Note: The rain and snow masks do not require strict alignment with the image.)

 - Prepare the dataset as follows:

```
./syn_data/
|--data
|  |--clear
|  |  |--000001.png
|  |  |--000002.png
|  |--depth_map
|  |  |--000001.png
|  |  |--000002.png
|  |--light_map
|  |  |--000001.png
|  |  |--000002.png
|  |--rain_mask
|  |  |--aaaaaa.png
|  |  |--bbbbbb.png
|  |--snow_mask
|  |  |--cccccc.png
|  |  |--dddddd.png
|--out
```

**2. Generate composite degradation images**

 - low+haze+rain

```
python syn_data.py --hq-file ./data/clear/ --light-file ./data/light_map/ --depth-file ./data/depth_map/ --rain-file ./data/rain_mask/ --snow-file ./data/snow_mask/ --out-file ./out/ --low --haze --rain
```

 - low+haze+snow

```
python syn_data.py --hq-file ./data/clear/ --light-file ./data/light_map/ --depth-file ./data/depth_map/ --rain-file ./data/rain_mask/ --snow-file ./data/snow_mask/ --out-file ./out/ --low --haze --snow
```
(Note: The degradation types can be customized according to specific needs.)

| Clear Image | low+haze+rain | low+haze+snow
| :--- | :--- | :---
| <img src="https://github.com/gy65896/OneRestore/blob/main/img_file/clear_img.jpg" width="200"> | <img src="https://github.com/gy65896/OneRestore/blob/main/img_file/l+h+r.jpg" width="200"> | <img src="https://github.com/gy65896/OneRestore/blob/main/img_file/l+h+s.jpg" width="200">

## Performance

### CDD-11

| Types             | Methods                                       | Venue & Year | PSNR ↑   | SSIM ↑   | #Params   |
|-------------------|-----------------------------------------------|--------------|----------|----------|------------|
| Input             | [Input](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMuNlQAAAAABf9KaFodlfC8H-K_MNiriFw?e=SiOrWU)                                         |              | 16.00    | 0.6008   | -          |
| One-to-One        | [MIRNet](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMuMlQAAAAABBzDLjLu69noXflImQ2V9ng?e=4wohVK)                                        | ECCV2020     | 25.97    | 0.8474   | 31.79M     |
| One-to-One        | [MPRNet](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMuLlQAAAAAB_iz3hjLHZDMi-RyxHKgDDg?e=SwSQML)                                        | CVPR2021     | 25.47    | 0.8555   | 15.74M     |
| One-to-One        | [MIRNetv2](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMuQlQAAAAAB2miyepdTE3qdy4z2-LM4pg?e=moXVAR)                                      | TPAMI2022    | 25.37    | 0.8335   | 5.86M      |
| One-to-One        | [Restormer](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMuPlQAAAAABE86t03kpAVm_TZDIBPKolw?e=vHAR7A)                                     | CVPR2022     | 26.99    | 0.8646   | 26.13M     |
| One-to-One        | [DGUNet](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMuOlQAAAAABZkHj8tMamqaGhQ0w4VwFrg?e=lfDUlx)                                        | CVPR2022     | 26.92    | 0.8559   | 17.33M     |
| One-to-One        | [NAFNet](https://1drv.ms/u/c/cbb69e4e3408ebcd/EWm9jiJiZLlLgq1trYO67EsB42LrjGpepvpS4oLqKnj8xg?e=5Efa4W)                                        | ECCV2022     | 24.13    | 0.7964   | 17.11M     |
| One-to-One        | [SRUDC](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMuWlQAAAAABf9RNAUZH_xL6wF4aODWKqA?e=h4EqVN)                                         | ICCV2023     | 27.64    | 0.8600   | 6.80M      |
| One-to-One        | [Fourmer](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMuXlQAAAAABQKrbA47G8kMD2cf7Chq5EQ?e=vOiWV0)                                       | ICML2023     | 23.44    | 0.7885   | 0.55M      |
| One-to-One        | [OKNet](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMuVlQAAAAABSMzfS1xEOxLeuvw8HsGyMw?e=jRmf9t)                                         | AAAI2024     | 26.33    | 0.8605   | 4.72M      |
| One-to-Many       | [AirNet](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMualQAAAAABYJ96PX0fipkP93zRXN_NVw?e=sXFOl8)                                        | CVPR2022     | 23.75    | 0.8140   | 8.93M      |
| One-to-Many       | [TransWeather](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMuZlQAAAAABoBiLjwJ8L2kl6rGQO5PeJA?e=msprhI)                                  | CVPR2022     | 23.13    | 0.7810   | 21.90M     |
| One-to-Many       | [WeatherDiff](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMuYlQAAAAABxdWbznZA1CQ0Bh1JH_ze-A?e=LEkcZw)                                   | TPAMI2023    | 22.49    | 0.7985   | 82.96M     |
| One-to-Many       | [PromptIR](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMublQAAAAAB9aGo3QK-WlKkL5ItITW9Hg?e=wXrJf1)                                      | NIPS2023     | 25.90    | 0.8499   | 38.45M     |
| One-to-Many       | [WGWSNet](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMudlQAAAAABi3HUMldxdoLHgDcUNoWMPw?e=z0qjAH)                                       | CVPR2023     | 26.96    | 0.8626   | 25.76M     |
| One-to-Composite  | [OneRestore](https://1drv.ms/u/c/cbb69e4e3408ebcd/Ec3rCDROnrYggMuclQAAAAABSmNvDBKR1u5rDtqQnZ8X7A?e=OcnrjY)                                    | ECCV2024     | 28.47    | 0.8784   | 5.98M      |
| One-to-Composite  | [OneRestore<sup>† </sup>](https://1drv.ms/u/c/cbb69e4e3408ebcd/EVM43y_W_WxAjrZqZdK9sfoBk1vpSzKilG0m7T-3i3la-A?e=dbNsD3)                          | ECCV2024     | 28.72    | 0.8821   | 5.98M      |

[Indicator calculation code](https://github.com/gy65896/OneRestore/blob/main/img_file/cal_psnr_ssim.py) and [numerical results](https://github.com/gy65896/OneRestore/blob/main/img_file/metrics_CDD-11_psnr_ssim.xlsx) can be download here.

</div>
<div align=center>
<img src="https://github.com/gy65896/OneRestore/blob/main/img_file/syn.jpg" width="1080">
</div>

### Real Scene

</div>
<div align=center>
<img src="https://github.com/gy65896/OneRestore/blob/main/img_file/real.jpg" width="1080">
</div>

### Controllability

</div>
<div align=center>
<img src="https://github.com/gy65896/OneRestore/blob/main/img_file/control1.jpg" width="410"><img src="https://github.com/gy65896/OneRestore/blob/main/img_file/control2.jpg" width="410">
</div>


## Citation

```
@inproceedings{guo2024onerestore,
  title={OneRestore: A Universal Restoration Framework for Composite Degradation},
  author={Guo, Yu and Gao, Yuan and Lu, Yuxu and Liu, Ryan Wen and He, Shengfeng},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```

#### If you have any questions, please get in touch with me (guoyu65896@gmail.com).
