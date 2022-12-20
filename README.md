
# CAT
This is an official PyTorch Implementation of **Towards Global Video Scene Segmentation with Context-Aware Transformer(CAT)** 

## 1. Environmental Setup
We have tested the implementation on the following environment:
  * Python 3.7.7 / PyTorch 1.7.1 / torchvision 0.8.2 / CUDA 11.0 / Ubuntu 18.04   

Also, the code is based on [pytorch-lightning](https://www.pytorchlightning.ai/) (==1.3.8) and all necessary dependencies can be installed by running following command. 
```bash
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install -r requirements.txt

# (optional) following installation of pillow-simd sometimes brings faster data loading.
$ pip uninstall pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

## 2. Prepare Data
Prepare data like [bassl](https://github.com/kakaobrain/bassl)

```bash
# <path-to-root>/CAT/data
movienet
│─ 240P_frames
│    │─ tt0120885                 # movie id (or video id)
│    │    │─ shot_0000_img_0.jpg
│    │    │─ shot_0000_img_1.jpg
│    │    │─ shot_0000_img_2.jpg  # for each shot, three key-frames are given.
|    |    :
│    :    │─ shot_1256_img_2.jpg
│    |    
│    │─ tt1093906
│         │─ shot_0000_img_0.jpg
│         │─ shot_0000_img_1.jpg
│         │─ shot_0000_img_2.jpg
|         :
│         │─ shot_1270_img_2.jpg
│
│─anno
     │─ anno.pretrain.ndjson
     │─ anno.trainvaltest.ndjson
     │─ anno.train.ndjson
     │─ anno.val.ndjson
     │─ anno.test.ndjson
     │─ vid2idx.json
│─scene318
     │─ label318
     │─ meta
     │─ shot_movie318
```

## 3. Train (Pre-training and Fine-tuning)
We use [Hydra](https://github.com/facebookresearch/hydra) to provide flexible training configurations.
Below examples explain how to modify each training parameter for your use cases.  
We assume that you are in `<path-to-root>` (i.e., root of this repository).  


### 3.1. Pre-training

** Pre-training CAT**  
Our pre-training is based on distributed environment (multi-GPUs training) using [ddp environment supported by pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html).  
The default setting requires 8-GPUs (of V100) with a batch of 256. However, you can set the parameter `config.DISTRIBUTED.NUM_PROC_PER_NODE` to the number of gpus you can use or change `config.TRAIN.BATCH_SIZE.effective_batch_size`.

```bash
cd <path-to-root>/CAT
EXPR_NAME=CAT_visual
WORK_DIR=$(pwd)
PYTHONPATH=${WORK_DIR} python3 ${WORK_DIR}/pretrain/main.py \
    config.EXPR_NAME=${EXPR_NAME} \
    config.DISTRIBUTED.NUM_NODES=1 \
    config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
    config.TRAIN.BATCH_SIZE.effective_batch_size=256
```
Note that the checkpoints are automatically saved in `bassl/pretrain/ckpt/<EXPR_NAME>` and log files (e.g., tensorboard) are saved in `bassl/pretrain/logs/<EXPR_NAME>.


### 3.2. Fine-tuning  

**(1) Extracting shot-level features from shot key-frames**    
For computational efficiency, we pre-extract shot-level representation and then fine-tune pre-trained models.  
Set `LOAD_FROM` to `EXPR_NAME` used in the pre-training stage and change `config.DISTRIBUTED.NUM_PROC_PER_NODE` as the number of GPUs you can use.
Then, the extracted shot-level features are saved in `<path-to-root>/bassl/data/movienet/features/<LOAD_FROM>`.

```bash
cd <path-to-root>/CAT
LOAD_FROM=CAT_visual
WORK_DIR=$(pwd)
PYTHONPATH=${WORK_DIR} python3 ${WORK_DIR}/pretrain/extract_shot_repr.py \
	config.DISTRIBUTED.NUM_NODES=1 \
	config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
	+config.LOAD_FROM=${LOAD_FROM}
```

**(2) Fine-tuning and evaluation**

```bash
cd <path-to-root>/CAT
WORK_DIR=$(pwd)
VISUAL_PRETRAINED_LOAD_FROM=CAT_visual
EXPR_NAME=transfer_finetune_${VISUAL_PRETRAINED_LOAD_FROM}
PYTHONPATH=${WORK_DIR} python3 ${WORK_DIR}/finetune/main.py \
	config.TRAIN.BATCH_SIZE.effective_batch_size=1024 \
	config.EXPR_NAME=${EXPR_NAME} \
	config.DISTRIBUTED.NUM_NODES=1 \
	config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
	config.TRAIN.OPTIMIZER.lr.base_lr=0.0000025 \
	+config.VISUAL_PRETRAINED_LOAD_FROM=${VISUAL_PRETRAINED_LOAD_FROM}
```




## 4. Citation
If you find this code helpful for your research, please cite our paper.
```
```

