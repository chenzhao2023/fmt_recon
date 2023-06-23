### 
This repo includes the data preparing code, model training, evaluation and results visualization code for paper FMT-reconstruction using shape priors with deformation network


### 0. Running environment

`tensorflow==1.15`

`numpy==1.18.0`

`scikit-image==0.17.1`

`seaborn==0.12.2`

`pyradiomics==2.2`

### 1. Data Generation

run data_prepare/MCX/train_data_create_sim.m

01_processing_data_MCX.py to conver mat files to numpy files

02_extract_features.py, use pyradiomics to generate features

03_clustering.py, use generated features to perform clustering

04_train_cubic_recon_single.py, train model with different number of clusters and training samples


### 2. training
training command:
python 04_train_cubic_recon_single.py --train=train --gpu=0 --n_cluster=4 --num_samples=1000 --cv=0
python 04_train_cubic_recon_single.py --train=train --gpu=0 --n_cluster=8 --num_samples=1000 --cv=0
python 04_train_cubic_recon_single.py --train=train --gpu=0 --n_cluster=12 --num_samples=1000 --cv=0
python 04_train_cubic_recon_single.py --train=train --gpu=0 --n_cluster=16 --num_samples=1000 --cv=0
python 04_train_cubic_recon_single.py --train=train --gpu=0 --n_cluster=20 --num_samples=1000 --cv=0
python 04_train_cubic_recon_single.py --train=train --gpu=0 --n_cluster=24 --num_samples=1000 --cv=0
python 04_train_cubic_recon_single.py --train=train --gpu=0 --n_cluster=28 --num_samples=1000 --cv=0

### 3. test and robust test
test commands:

K=4
python 04_train_cubic_recon_single.py --train=test --gpu=0 --model_path=/media/z/data3/wei/exp_single/cv0_k4_size64_1000/0050/model.cpkt --n_cluster=4 --num_samples=1000 --cv=0

python 04_train_cubic_recon_single.py --train=robust_test --sigma=-1 --gpu=0 --model_path=/media/z/data3/wei/exp_single/cv0_k4_size64_1000/0050/model.cpkt --n_cluster=4 --num_samples=1000 --cv0

K=8
python 04_train_cubic_recon_single.py --train=test --gpu=0 --model_path=/media/z/data3/wei/exp_single/cv0_k8_size64_1000/0050/model.cpkt --n_cluster=8 --num_samples=1000 --cv=0

python 04_train_cubic_recon_single.py --train=robust_test --sigma=-1 --gpu=0 --model_path=/media/z/data3/wei/exp_single/cv0_k8_size64_1000/0050/model.cpkt --n_cluster=8 --num_samples=1000 --cv0

K=12
python 04_train_cubic_recon_single.py --train=test --gpu=0 --model_path=/media/z/data3/wei/exp_single/cv0_k12_size64_1000/0200/model.cpkt --n_cluster=12 --num_samples=1000 --cv=0

python 04_train_cubic_recon_single.py --train=robust_test --sigma=-1 --gpu=0 --model_path=/media/z/data3/wei/exp_single/cv0_k12_size64_1000/0200/model.cpkt --n_cluster=12 --num_samples=1000 --cv0

K=16
python 04_train_cubic_recon_single.py --train=test --gpu=0 --model_path=/media/z/data3/wei/exp_single/cv0_k16_size64_1000/0200/model.cpkt --n_cluster=16 --num_samples=1000 --cv=0

python 04_train_cubic_recon_single.py --train=robust_test --sigma=-1 --gpu=0 --model_path=/media/z/data3/wei/exp_single/cv0_k16_size64_1000/0200/model.cpkt --n_cluster=16 --num_samples=1000 --cv0

k= 20
test: 
python 04_train_cubic_recon_single.py --train=test --gpu=0 --model_path=/media/z/data3/wei/exp_single/cv0_k20_size64_1000/0160/model.cpkt --n_cluster=20 --num_samples=1000 --cv=0

test with random noise: 
python 04_train_cubic_recon_single.py --train=robust_test --sigma=1 --gpu=0 --model_path=/media/z/data3/wei/exp_single/cv0_k20_size64_1000/0160/model.cpkt --n_cluster=20 --num_samples=1000 --cv=0

k= 24
test: 
python 04_train_cubic_recon_single.py --train=test --gpu=0 --model_path=/media/z/data3/wei/exp_single/cv0_k24_size64_1000/0180/model.cpkt --n_cluster=24 --num_samples=1000 --cv=0

test with random noise: 
python 04_train_cubic_recon_single.py --train=robust_test --sigma=1 --gpu=0 --model_path=/media/z/data3/wei/exp_single/cv0_k24_size64_1000/0180/model.cpkt --n_cluster=24 --num_samples=1000 --cv=0

### 4. test dual source reconstruction
python 04_train_cubic_recon_single.py --train=dual --gpu=0 --model_path=/media/z/data3/wei/exp_single/cv0_k20_size64_1000/0160/model.cpkt --n_cluster=20 --num_samples=1000 --cv=0


### 5. V-Net
train:
python 04_train_vnet_single.py --train=train --gpu=0 --cv=0 --num_samples=1000

test:
python 04_train_vnet_single.py --train=test --gpu=0 --cv=0 --num_samples=1000 --model_path=/media/z/data3/wei/exp_vnet/cv0_size64_vnet_1000/0140/model.cpkt
python 04_train_vnet_single.py --train=robust_test --gpu=0 --cv=0 --sigma=-1 --num_samples=1000 --model_path=/media/z/data3/wei/exp_vnet/cv0_size64_vnet_1000/0140/model.cpkt


