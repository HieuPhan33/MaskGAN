# MaskGAN for Unpaired MR-to-CT Translation [MICCAI2023]

 [![arXiv](https://img.shields.io/badge/arXiv-2311.12437-blue)](https://arxiv.org/pdf/2307.16143) [![cite](https://img.shields.io/badge/cite-BibTex-yellow)](cite.bib)

## 📢 Updates!

* New publication is coming out! **Mixed-view Refinement MaskGAN: Anatomical Preservation for Unpaired MRI-to-CT Synthesis**. Stay tuned!
* **Refinement MaskGAN version**: An extended model, which refined images using a simple, yet effective multi-stage, multi-plane approach is develop to improve the volumetric definition of synthetic images.
* **Model enhancements**: We include selection strategies to choose similar MRI/CT matches based on the position of slices.

## 🏆 MaskGAN Framework

A novel unsupervised MR-to-CT synthesis method that:
- Preserves the anatomy under the explicit supervision of coarse masks without using costly manual annotations. MaskGAN bypasses the need for precise annotations, replacing them with standard (unsupervised) image processing techniques, which can produce coarse anatomical masks.
- Introduces a Shape consistency loss to preserve the overall structure of images after a cycle of translation.

![Framework](./imgs/maskgan_v2.svg)

## Comparison with State-of-the-Art Methods on Paediatric MR-CT Synthesis
![Result](./imgs/results.jpg)


The repository offers the official implementation of our paper in PyTorch. The next reflects some results of using **MaskGAN** in different benchmarks. 


<table>
    <thead>
        <tr>
            <th style="width: 10px;">Data</th>
            <th>Sample</th>
            <th>MAE</th>
            <th>SSIM</th>
            <th>PSNR</th>
            <th>Weights</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Unpaired Pediatric brain MRI/CT images (Private dataset)</td>
            <td><img src="imgs/pediatric_mri_sample.png" width="100"> <img src="imgs/pediatric_ct_sample.png" width="100"></td>
            <td>60.76</td>
            <td>83.23</td>
            <td>23.95</td>
            <td><a href="https://drive.google.com/file/d/15e1pS2V2DDdQQqIdEdD7cpZstyQuSG_i/view?usp=drive_link">Download</a></td>
        </tr>
        <tr>
            <td>Unpaired Adult brain MRI/CT images. Original dataset are paired (<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8446124/">Link</a>)</td>
            <td><img src="imgs/adult_mri_sample.png" width="100"> <img src="imgs/adult_ct_sample.png" width="100"></td>
            <td>44.30</td>
            <td>87.10</td>
            <td>22.56</td>
            <td><a href="https://drive.google.com/file/d/1FUTEDrw8G92zgc0rRZ4TRxHFgkkPhk7R/view?usp=drive_link">Download</a></td>
        </tr>
        <tr>
            <td>Unpaired Adult abdominal MRI/CT images (<a href="https://chaos.grand-challenge.org/Download/">Link</a>)</td>
            <td><img src="imgs/abdominal_mri_sample.png" width="100" style="transform: rotate(270deg);"> <img src="imgs/abdominal_ct_sample.png" width="100" style="transform: rotate(270deg);"></td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td><a href="https://drive.google.com/file/d/1pa3vsPohB6_aCmGoHe_1V851yZHg_1Ae/view?usp=drive_link">Download</a></td>
        </tr>
    </tbody>
</table>



## 🛠️ Installation
### Option 1: Directly use our Docker image
- We have created a public docker image `stevephan46/maskgan:d20b79d4731210c9d287a370e37b423006fd1425`.
- Script to pull docker image and run docker container for environment setup:
```bash
docker pull stevephan46/maskgan:d20b79d4731210c9d287a370e37b423006fd1425
docker run --name maskgan --gpus all --shm-size=16g -it -v /path/to/data/root:/data stevephan46/maskgan:d20b79d4731210c9d287a370e37b423006fd1425
```
- Mount the folder storing your data folder in `-v /path/to/data/root:/data`.
- In the docker container, clone the code and follow next following steps.

### Option 2: Install environments locally

- This code uses PyTorch 1.8.1, Python 3.8 and [apex](https://github.com/NVIDIA/apex) for half-precision training support.

- Please install PyTorch and [apex](https://github.com/NVIDIA/apex), then install other dependencies by
```bash
pip install -r requirements.txt
```

## 📚 Dataset Preparation and Mask Generations
Refer to [preprocess/README.md](./preprocess/README.md) file.

## 🚀 MaskGAN Training and Testing
- Sampled training script is provided in train.sh
- Modify image augmentations as needed `--load_size` (resize one dimension to be a fixed size), `--pad_size` (pad both dimensions to an equal size), `--crop_size` (crop both dimensions to an equal size).
- Train a model:
```
sh train.sh
```
- `lambda_mask` and `lambda_shape` specify hyper-parameters of our proposed mask loss and shape consistency loss.
- `opt_level` specifies Apex mixed-precision optimization level. The default is `O0` which is full FP32 training. If low GPU memory, you can use O1 or O2 for mixed precision training.
- Training command:
```
python train.py --dataroot dataroot --name exp_name --gpu_ids 0 --display_id 0 --model mask_gan --netG att 
--dataset_mode unaligned --pool_size 50 --no_dropout
--norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --lambda_mask 1.0 --lambda_shape 0.5 --load_size 150 --pad_size 225 --crop_size 224 --preprocess resize_pad_crop --no_flip
--batch_size 4 --niter 40 --niter_decay 40 --display_freq 1000 --print_freq 1000 --n_attentions 5
```
- For your own experiments, you might want to specify --netG, --norm. Our mask generators `netG` are `att` and `unet_att`.
- To continue model training, append `--continue_train --epoch_count xxx` on the command line.
- Test the model:
```
sh test.sh
```
- The results will be saved at `./results/exp_name`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory. There will be four folders `fake_A`, `fake_B`, `real_A`, `real_B` created in `results`.

## 💾 Use of pretrained weights

This [zip file](https://drive.usercontent.google.com/download?id=15e1pS2V2DDdQQqIdEdD7cpZstyQuSG_i&export=download) contains trained weights of MaskGAN run over MRI/CT pediatric brain dataset. To use them, unzip the contents in the folder `pretrained_weights`. You can use them as pretrained weights during your training step or using directly for testing with the defaults parameters. Just add the next parameter.

```
--use_pretrained_weights True
```

## 🔍 Evaluate results
- The script `eval.sh` is provided as an example. Modify the variable `exp_name` to match your experiment name specified by parameter `--name` when running test.py.

## 📜 Citation
If you use this code for your research, please cite our papers.

```
@inproceedings{phan2023maskgan,
  title = {Structure-Preserving Synthesis: {MaskGAN} for Unpaired {MR-CT} Translation},
  author = {Phan, Minh Hieu and Liao, Zhibin and Verjans, Johan and To, Minh-Son},
  booktitle = {International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year = {2023}
}
```

## 🙏 Acknowledgments
This source code is inspired by [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [AttentionGAN](https://github.com/Ha0Tang/AttentionGAN). 

