 # <p align=center> [ECCV 2024] OneRestore: A Universal Restoration Framework for Composite Degradation</p>

<div align="center">
 
[![arXiv](https://img.shields.io/badge/OneRestore-arXiv-red.svg)](https://arxiv.org/abs/2407.04621)
[![Web](https://img.shields.io/badge/OneRestore-Web-blue.svg)](https://gy65896.github.io/Projects/OneRestore/index.html)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgy65896%2FOneRestore&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>
<div align=center>
<img src="https://github.com/gy65896/OneRestore/assets/48637474/7e037f8e-8a8d-4953-8aa6-5142e64f2005" width="720">
</div>

---
>**OneRestore: A Universal Restoration Framework for Composite Degradation**<br>  [Yu Guo](https://scholar.google.com/citations?user=klYz-acAAAAJ&hl=zh-CN)<sup>†</sup>, [Yuan Gao](https://scholar.google.com.hk/citations?hl=zh-CN&user=4JpRnU4AAAAJ&view_op=list_works&sortby=pubdate)<sup>†</sup>, [Yuxu Lu](https://scholar.google.com.hk/citations?user=XXge2_0AAAAJ&hl=zh-CN), [Huilin Zhu](https://scholar.google.com.hk/citations?hl=zh-CN&user=fluPrxcAAAAJ), [Ryan Wen Liu](http://mipc.whut.edu.cn/index.html)<sup>* </sup>, [Shengfeng He](http://www.shengfenghe.com/)<sup>* </sup> <br>
(† Equal Contribution, * Corresponding Author)<br>
>European Conference on Computer Vision

> **Abstract:** *In real-world scenarios, image impairments often manifest as composite degradations, presenting a complex interplay of elements such as low light, haze, rain, and snow. Despite this reality, existing restoration methods typically target isolated degradation types, thereby falling short in environments where multiple degrading factors coexist. To bridge this gap, our study proposes a versatile imaging model that consolidates four physical corruption paradigms to accurately represent complex, composite degradation scenarios. In this context, we propose OneRestore, a novel transformer-based framework designed for adaptive, controllable scene restoration. The proposed framework leverages a unique cross-attention mechanism, merging degraded scene descriptors with image features, allowing for nuanced restoration. Our model allows versatile input scene descriptors, ranging from manual text embeddings to automatic extractions based on visual attributes. Our methodology is further enhanced through a composite degradation restoration loss, using extra degraded images as negative samples to fortify model constraints. Comparative results on synthetic and real-world datasets demonstrate OneRestore as a superior solution, significantly advancing the state-of-the-art in addressing complex, composite degradations.*
---

## News 🚀
* **2024.07.07**: Code and Datasets are released.
* **2024.07.02**: OneRestore is accepted by ECCV2024.

## Network Architecture

</div>
<div align=center>
<img src="https://github.com/gy65896/OneRestore/assets/48637474/e26fcaae-3688-489f-8bb4-a698bae3e7fb" width="1080">
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
pip install genism
```

### Pretrained Models

Please download our pre-trained models and put them in  `./ckpts`.

| Model | Description
| :--- | :----------
|[embedder_model.tar](https://1drv.ms/u/s!As3rCDROnrbLgqpnhSQFIoD9msXWOA?e=aUpHOT)  | Text/Visual Embedder trained on our CDD-11.
|[onerestore_cdd-11.tar](https://1drv.ms/u/s!As3rCDROnrbLgqpmWkGBku6oj33efg?e=7yUGfN)  | OneRestore trained on our CDD-11.
|[onerestore_real.tar](https://1drv.ms/u/s!As3rCDROnrbLgqpi-iJOyN6OSYqiaA?e=QFfMeL)  | OneRestore trained on our CDD-11 for Real Scenes.
|[onerestore_lol.tar](https://1drv.ms/u/s!As3rCDROnrbLgqpkSoVB1j-wYHFpHg?e=0gR9pn)  | OneRestore trained on LOL (low light enhancement benchmark).
|[onerestore_reside_ots.tar](https://1drv.ms/u/s!As3rCDROnrbLgqpjGh8KjfM_QIJzEw?e=zabGTw)  | OneRestore trained on RESIDE-OTS (image dehazing benchmark).
|[onerestore_rain1200.tar](https://1drv.ms/u/s!As3rCDROnrbLgqplAFHv6B348jarGA?e=GuduMT)  | OneRestore trained on Rain1200 (image deraining benchmark).
|[onerestore_snow100k.tar](https://1drv.ms/u/s!As3rCDROnrbLgqphsWWxLZN_7JFJDQ?e=pqezzo)  | OneRestore trained on Snow100k-L (image desnowing benchmark).

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

We provide the download link of our Composite Degradation Dataset with 11 types of degradation ([CDD-11](https://1drv.ms/f/s!As3rCDROnrbLgqpezG4sao-u9ddDhw?e=A0REHx)).

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

1. Train Text/Visual Embedder by

```
python train_Embedder.py --train-dir ./data/CDD-11_train --test-dir ./data/CDD-11_test --check-dir ./ckpts --batch 256 --num-workers 0 --epoch 200 --lr 1e-4 --lr-decay 50
```

2. Remove the optimizer weights in the Embedder model file by

```
python remove_optim.py --type Embedder --input-file ./ckpts/embedder_model.tar --output-file ./ckpts/embedder_model.tar
```

3. Generate the `dataset.h5` file for training OneRestore by

```
python makedataset.py --train-path ./data/CDD-11_train --data-name dataset.h5 --patch-size 256 --stride 200
```

4. Train OneRestore model by

```
python train_OneRestore.py --embedder-model-path ./ckpts/embedder_model.tar --save-model-path ./ckpts --train-input ./dataset.h5 --test-input ./data/CDD-11_test --output ./result/ --epoch 120 --bs 4 --lr 1e-4 --adjust-lr 30 --num-works 4
```

5. Remove the optimizer weights in the OneRestore model file by

```
python remove_optim.py --type OneRestore --input-file ./ckpts/onerestore_model.tar --output-file ./ckpts/onerestore_model.tar
```

## Performance

### CDD-11

</div>
<div align=center>
<img src="https://github.com/gy65896/OneRestore/assets/48637474/e8b5d6f6-b78c-43a8-9c21-4e78c166fecf" width="720">
</div>

</div>
<div align=center>
<img src="https://github.com/gy65896/OneRestore/assets/48637474/835edc0c-acfb-481c-9116-a23ce1929588" width="1080">
</div>

### Real Scene

</div>
<div align=center>
<img src="https://github.com/gy65896/OneRestore/assets/48637474/f9a4df1c-ad64-4339-8485-b76f29010bdd" width="1080">
</div>

### Controllability

</div>
<div align=center>
<img src="https://github.com/gy65896/OneRestore/assets/48637474/ed57114a-43a5-4221-bc3a-9bc7f3ac2dd5" width="410"><img src="https://github.com/gy65896/OneRestore/assets/48637474/fd8684f7-8494-4fba-8919-dc50e6acb26f" width="410">
</div>

## Citation

```
@inproceedings{guo2024onerestore,
  title={OneRestore: A Universal Restoration Framework for Composite Degradation},
  author={Guo, Yu and Gao, Yuan and Lu, Yuxu and Liu, Ryan Wen and He, Shengfeng},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2024}
}
```

#### If you have any questions, please get in touch with me (yuguo@whut.edu.cn).
