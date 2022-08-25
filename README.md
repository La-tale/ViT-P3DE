# PaperID 6429
This repository contains the code of "ViT-P3DE\*: Vision Transformer Based Multi-Camera Instance Association with Pseudo 3D Position Embeddings".

Due to the AAAI2023 review policy, external links (except the publicly availble links), some codes and explanations are ommited here.

If our paper is accepted, full version of this repository will be uploaded to open-source community with missing external links, more detailed explanation and some additional codes. 

Some of our codes are based on the following repositories: [MessyTable](https://github.com/caizhongang/MessyTable), [DeiT](https://github.com/facebookresearch/deit), and [TransReID](https://github.com/damo-cv/TransReID).

We'd like to thank the authors providing the codes.

## 1. Setup
### Environment
We use PyTorch v1.8 and torchvision v0.9 with a single NVIDIA A100 GPU.

Python 3.7 is needed to use the KMSolver module provided in "src" folder.

We test our code with ubuntu 18.04, nvidia-driver v510.47, and cuda v11.1.
#### 1) Setup dependencies
~~~bash
conda create -n vit_p3de python=3.7
conda activate vit_p3de
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install opencv-python==3.4.2.17
pip install scipy==1.2.0
pip install sklearn==0.0
pip install timm==0.5.4
pip install PyYAML==5.4.1
~~~

### Dataset preparation
We cite the preparation guideline from [MessyTable](https://github.com/caizhongang/MessyTable).

The project homepage of MessyTabe is in [MessyTable](https://github.com/caizhongang/MessyTable).
* Download MessyTable.zip (~22 GB) from [[Aliyun]](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/MessyTable.zip) or [[Google Drive]](https://drive.google.com/file/d/1i4mJz9xsDwhzWes7sVLXuhLKP9eNtbBG/view?usp=sharing)
* Unzip MessyTable.zip, check the unzipped folder includes `images/` and `labels/`
* Rename the unzipped folder to `data/`, place `data/` in this repository as follows:

```bash
ViT-P3DE
├──data
   ├── images
   └── labels
``` 
### Download pre-trained models
* **Download link will be available if paper is accepted**.

## 2. Training
Run the command below to train the framework. For more options, please refer to `src/train.py`

Default configuration is training on 1 GPUs with one batch size (64 triplet pairs)

Example for ViT-P3DE\*
~~~bash
python3 src/train.py --config_dir tripletnet_1gpu_all_pairs_vit_tiny_p3de
~~~

Arguments:
* --config_dir: the directory that contains the specific config file `train.yaml` (checkpoints are automatically saved in the same dir)

## 3. Evaluation
### Evaluation on MessyTable 
To reproduce the experiments in our paper, please use the provided pretrained weights.

For detailed command options, please refer to `src/test.py`

Example for ViT-P3DE\*

~~~bash
python3 src/test.py --config_dir tripletnet_1gpu_all_pairs_vit_tiny_p3de \
--eval_json test.json \
--save_features \
--eval_model
~~~

Arguments:
* --config_dir: the directory that contains the specific config file `train.yaml` (checkpoints are automatically saved in the same dir)
* --eval_json: data split name in `data/labels/` to evaluate test.json, val.json
* --save_features: (optional) save extrated features in `models/<config_dir>` for faster evaluation in the future
* --load_features: (optional) load saved features from `models/<config_dir>`, if the features have been saved in the past
* --eval_model: evaluate using the appearance features only
* --eval_model_esc: (optional) evaluate using the appearance features with epipolar soft constraint (ESC)
* --eval_by_angle: (optional) evaluate by angle differences
* --eval_by_subncls: (optional) evaluate superclass-single (SPS), superclass-duplicate (SPD), subclass-single (SBS), and subclass-duplicate (SBD)

### Evaluation in terms of each constraint (Sec 4.4)
We present the commands that we use when we evaluate the frameworks with following constraints.

To evaluate the frameworks with following constraints, the features extracted from evaluation with "eval_model" option are necessary.

Please run these commands after the evaluation command with "eval_model" and "save_features" options.
#### 1) Appearance differences among identical objects due to angle variation
Example for ViT-P3DE\*
~~~bash
python3 src/test.py --config_dir tripletnet_1gpu_all_pairs_vit_tiny_p3de \
--eval_json test.json \
--load_features \
--eval_by_angle
~~~
#### 2) Presence of similar objects in one scene
Example for ViT-P3DE\*
~~~bash
python3 src/test.py --config_dir tripletnet_1gpu_all_pairs_vit_tiny_p3de \
--eval_json test.json \
--load_features \
--eval_by_subncls
~~~
#### 3) Object occlusion
Example for ViT-P3DE\*

Unlike aformentioned constraints, we evaluate instance-pairs in each test json file.

To avoid evaluations on three files, we implement python file ('find_index.py') to extract features corresponding to each json file.

You don't need to run "find_index.py" after running it once.
~~~bash
python3 find_index.py models/tripletnet_1gpu_all_pairs_vit_tiny_p3de/

python3 src/test.py --config_dir tripletnet_1gpu_all_pairs_vit_tiny_p3de \
--eval_json test_easy.json \
--load_features \
--eval_by_subncls

python3 src/test.py --config_dir tripletnet_1gpu_all_pairs_vit_tiny_p3de \
--eval_json test_medium.json \
--load_features \
--eval_by_subncls

python3 src/test.py --config_dir tripletnet_1gpu_all_pairs_vit_tiny_p3de \
--eval_json test_hard.json \
--load_features \
--eval_by_subncls
~~~
