# PointCMT
[![arXiv](https://img.shields.io/badge/arXiv-2203.09065-b31b1b.svg)](https://arxiv.org/pdf/2210.04208.pdf)
[![GitHub Stars](https://img.shields.io/github/stars/ZhanHeshen/PointCMT?style=social)](https://github.com/ZhanHeshen/PointCMT)
![visitors](https://visitor-badge.glitch.me/badge?page_id=https://github.com/ZhanHeshen/PointCMT)

This repository is for **PointCMT** introduced in the following paper

[Xu Yan*](https://yanx27.github.io/), Heshen Zhan*, Chaoda Zheng, Jiantao Gao, Ruimao Zhang, Shuguang Cui, 
[Zhen Li*](https://mypage.cuhk.edu.cn/academics/lizhen/), 
"*Let Images Give You More: Point Cloud Cross-Modal Training for Shape Analysis*", NeurIPS 2022 (**Spotlight**) :smiley: [[arxiv]](https://arxiv.org/pdf/2210.04208.pdf).

![image](figures/pipeline.jpg)

If you find our work useful in your research, please consider citing:
```latex
@InProceedings{yan2022let,
      title={Let Images Give You More: Point Cloud Cross-Modal Training for Shape Analysis}, 
      author={Xu Yan and Heshen Zhan and Chaoda Zheng and Jiantao Gao and Ruimao Zhang and Shuguang Cui and Zhen Li},
      year={2022},
      booktitle={NeurIPS}
}

@inproceedings{yan20222dpass,
  title={2dpass: 2d priors assisted semantic segmentation on lidar point clouds},
  author={Yan, Xu and Gao, Jiantao and Zheng, Chaoda and Zheng, Chao and Zhang, Ruimao and Cui, Shuguang and Li, Zhen},
  booktitle={European Conference on Computer Vision},
  pages={677--695},
  year={2022},
  organization={Springer}
}
```
Our another work for cross-modal semantic segmentation (ECCV 2022) is released [HERE](https://github.com/yanx27/2DPASS).

## Update
**2022/12/19:** The codes for ModelNet40 are released :rocket:!<br>
**2022/11/16:** Our paper is selected as **spotlight** in NeurIPS 2022!

## Install
The latest codes are tested on CUDA10.1, PyTorch 1.4.0 and Python 3.7.5. Please use the same environment with us!
```shell
conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```
Compile the library through:
```shell script
cd emdloss
python setup.py install
cd ../pointnet2/
pip install -e . && cd ..
```

## Data Preparation
* Download alignment point cloud data of **ModelNet40** [HERE](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `dataset/ModelNet40/modelnet40_normal_resampled/`.
* Download multiview dataset [HERE](https://cuhko365-my.sharepoint.com/:u:/g/personal/220019191_link_cuhk_edu_cn/EVm3wAL4nQNIh397VdgMUS8BiHNeTLzC_TcTCi7akk1omA?e=G3WoS5)  and save in `dataset/ModelNet40/ModelNet40_mv_20view/`.
* Process the multiview and point clouds, and the data will be saved in `dataset/ModelNet40/data/`
```shell script
cd data
python modelnet40_processor.py
```
You can also download processed data directly from [HERE](https://cuhko365-my.sharepoint.com/:f:/g/personal/220019191_link_cuhk_edu_cn/Er6xagknF2tBqC5FVUsyVe4BzEiu45D23rAv5wS2xGdsqA?e=Ygzv6q).

## Training on ModelNet40
### Stage I
Train and evaluate MVCNN model through
```shell
# training
python main_mvcnn_modelnet.py --exp_name mvcnn_default

# evaluation
python main_mvcnn_modelnet.py --exp_name mvcnn_default --eval --model_path [CHECKPOINT_PATH]
```
Note that the large batch size for MVCNN is **necessary**!! If you use smaller batch size due to the memory limitation, you need increase the `--accumulation_step` during the training. After training, the performance of MVCNN should be around 97%.

### Stage II
Training cross-model point generator through
```shell
python train_cmpg.py --exp_name cmpg_default --teacher_path [MVCNN_CHECKPOINT]
```

### Stage III
Before conducting cross-modal training, we first generate the features of MVCNN offline to speedup the training:
```shell
python offline_feature_generation.py --model_path [MVCNN_CHECKPOINT]
```
The features will be saved in `dataset/ModelNet40/data/`.
After that, we train PointNet++ with PointCMT through
```shell
python train_pointcmt.py --exp_name pointnet2_pointcmt --cmpg_checkpoint [CMPG_CHECKPOINT]
```
You can also train the vanilla model through `--no_pointcmt` option.

## Inference
We conduct voting test on our trained model:
```shell
python voting_test.py --checkpoint [POINITNET2_CHECKPOINT]
```

## Model Zoo
You can also use our pre-trained models from `pretrained/modelnet40/`, including MVCNN, CMPG and PointNet++. More models and codes for ScanObjectNN will be released very soon!
|Model | Accuracy |
|:---:|:---:|
|MVCNN|97.0%|
|PointNet++|94.6%|

# Acknowledgements
Code is built based on [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch), [SimpleView](https://github.com/princeton-vl/SimpleView) and [EMD(Earth Mover's Distance)](https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd).

# License
This repository is released under MIT License (see LICENSE file for details).
