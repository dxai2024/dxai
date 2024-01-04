# DXAI: Explaining Classification by Image Decomposition
## Abstract
We propose a new way to explain and to visualize neural network classification through a decomposition-based explainable AI (DXAI).
Instead of providing an explanation heatmap, our method yields a decomposition of the image into class-agnostic and class-distinct parts, with respect to the data and chosen classifier. Following a fundamental signal processing paradigm of analysis and synthesis, the original image is the sum of the decomposed parts. We thus obtain a radically different way of explaining classification. The class-agnostic part ideally is composed of all image features which do not posses  class information, where the class-distinct part is its complementary.
This new perceptual visualization, can be more helpful and informative in certain scenarios, especially when the attributes are dense, global and additive in nature, for instance, when colors or textures are essential for class distinction.
![Heatmaps compare](https://github.com/dxaicvpr2024/DXAI/blob/main/heatmaps_compare.jpg)

## Installation
The code in this repository is written based on the stragan-v2 code that can be found [here](https://github.com/clovaai/stargan-v2) [[1]](#1).
Therefore the installation of the packages and the requirements are similar. To install:

```bash
git clone https://github.com/dxaicvpr2024/DXAI.git
cd DXAI/
conda create -n dxai python=3.6.7
conda activate dxai
pip install -r requirements.txt
```
The installations and experiments were conducted on a system equipped with an NVIDIA GeForce RTX 2080 Ti GPU, running CUDA version 12.0.

## Data
Download AFHQ dataset and pretrained models.
```bash
bash download.sh afhq-dataset
bash download.sh pretrained-resnet18-afhq
bash download.sh pretrained-dxai-afhq

```
Download CelebA-hq dataset and pretrained models.
```bash
bash download.sh celeba-hq-dataset
bash download.sh pretrained-resnet18-celeba-hq
bash download.sh pretrained-dxai-celeba-hq

```
## Evaluation
```bash
# afhq
python main.py --mode eval\
      --checkpoint_dir expr/checkpoints_afhq_pretrained \
      --data_name afhq\
      --mission_name pretrained\
      --use_pretrained_classifier 1 \
      --classifier_type resnet18\
      --num_branches 5\
      --img_channels 3\
      --img_size 256\
      --max_eval_iter 1500 &

# celeba-hq
python main.py --mode eval\
      --checkpoint_dir expr/checkpoints_celeba_hq_pretrained \
      --data_name celeba_hq\
      --mission_name pretrained\
      --use_pretrained_classifier 1 \
      --classifier_type resnet18\
      --num_branches 5\
      --img_channels 3\
      --img_size 256\
      --max_eval_iter 2000 &
```
The results including the graphs, their AUC and the distinction maps will be stored in `xai_output/`.
## Training
To train DXAI from scratch use the following command example or use [run_commands.py](run_commands.py). Generated images and network checkpoints will be stored in the `expr/samples` and `expr/checkpoints` directories, respectively.
```bash
# afhq
python main.py --mode train\
      --sample_dir expr/samples_afhq_training\
      --checkpoint_dir expr/checkpoints_afhq_training \
      --src_dir assets/afhq\
      --train_img_dir ../Data/afhq/train \
      --val_img_dir ../Data/afhq/val \
      --resume_iter 0 \
      --data_name afhq\
      --mission_name training\
      --use_pretrained_classifier 1 \
      --classifier_type resnet18\
      --num_branches 5\
      --img_channels 3\
      --img_size 256 --batch_size 2\
      --sample_every 2500 --save_every 10000 --total_iters 300001 &'
```
To understand the role of each argument, please look at [core/load_args.py](core/load_args.py).

## References
<a id="1">[1]</a> 
Yunjey Choi and Youngjung Uh and Jaejun Yoo and Jung-Woo Ha (2020). 
StarGAN v2: Diverse Image Synthesis for Multiple Domains
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
