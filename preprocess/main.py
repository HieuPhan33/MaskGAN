import numpy as np
import matplotlib.pyplot as plt
import ants
import pandas as pd
# import cv2
# import imageio
from PIL import Image
from skimage.measure import label   
from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                      morphology, restoration, segmentation, transform,
                      util)
import skimage
import os
import glob
import imageio
import argparse
from tqdm import tqdm


def visualize(img, filename, step=10):
    shapes = img.shape
    for i, shape in enumerate(shapes):
        fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(12,12))
        start = shape//2 - step*4
        for t, ax in enumerate(axes.flatten()):
            if i == 0:
                data = img[start + t*step, :, :]
            elif i == 1:
                data = img[:, start+t*step, :]
            else:
                data = img[:, :, start+t*step]
            ax.imshow(data, cmap='gray', origin='lower')
            ax.axis('off')
        fig.tight_layout()
        plt.savefig(f'{filename}_{i}.png')
        plt.clf()

def normalize(img, min_=None, max_=None):
    if min_ is None:
        min_ = img.min()
    if max_ is None:
        max_ = img.max()
    return (img - min_)/(max_ - min_)


def crop_scan(img, mask, crop=0, crop_h=0, ignore_zero=True):
    # Swap dimensions for visualizability - modify as needed
    img = np.transpose(img, (0,2,1))[:,::-1,::-1]
    if mask is not None:
        mask = np.transpose(mask, (0,2,1))[:,::-1,::-1]
    
    # Exclude all zero (air only) slices during training - modify as needed
    if ignore_zero:
        mask_ = img.sum(axis=(1,2)) > 0
        img = img[mask_]
        if mask is not None:
            mask = mask[mask_]
    
    # Crop depth dimension - modify as needed
    if crop > 0:
        length = img.shape[0]
        img = img[int(crop*length): int((1-crop)*length)]
        if mask is not None:
            mask = mask[int(crop*length): int((1-crop)*length)]
    
    # Crop height dimension - modify as needed
    if crop_h > 0:
        if img.shape[1] > 200:
            crop_h = 0.8
        new_h = int(crop_h*img.shape[1])
        img = img[:, :new_h]
        if mask is not None:
            mask = mask[:, :new_h]

    return img, mask

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC




def get_3d_mask(img, min_, max_=None, th=50, width=2):
    if max_ is None:
        max_ = img.max()
    img = np.clip(img, min_, max_)
    img = np.uint8(255*normalize(img, min_, max_))

    ## Remove artifacts
    mask = np.zeros(img.shape).astype(np.int32)
    mask[img > th] = 1

    ## Remove artifacts and small holes with binary opening
    mask = morphology.binary_opening(mask, )

    remove_holes = morphology.remove_small_holes(
        mask, 
        area_threshold=width ** 3
    )

    
    largest_cc = getLargestCC(remove_holes)
    return img, largest_cc.astype(np.int32)


def save_slice(img, mask, data_dir, data_mask_dir, filename):
    assert img.shape == mask.shape, f"Shape not match - img {img.shape} vs mask {mask.shape}"
    for i in range(len(img)):
        im = np.uint8(255*normalize(img[i])) 
        m = np.uint8(255*normalize(mask[i])) 
        imageio.imwrite(f'{data_dir}/{filename}_{i}.png', im)
        imageio.imwrite(f'{data_mask_dir}/{filename}_{i}.png', m)

def parse_float_array(s):
    try:
        values = [float(x.strip()) for x in s.split(',')]
        return values
    except ValueError:
        raise argparse.ArgumentTypeError('Invalid float array format. Example: 1.0, 1.0, 1.0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data')
    # Add arguments
    parser.add_argument('--root', type=str, help='Data root dir')
    parser.add_argument('--out', type=str, default='processed-mr-ct', help='Output directory')
    parser.add_argument('--resample', nargs='+', type=float, default=[1.0, 1.0, 1.0], help='Resample scan resolutions')
    args = parser.parse_args()

    data_dir = args.root
    out_dir = args.out
    resample = args.resample

    # Set clip CT intensity
    min_ct, max_ct = -1000, 2000
    th = 10 # Consider pixel less than certain threshold as background (remove noise, artifacts)

    # Modify pattern matching expression to list MR and CT nifti files
    root_a = f'{data_dir}/MRI/*.nii'
    root_b = f'{data_dir}/CT/*.nii'
    output_a_dir = f'{out_dir}/train_A'
    output_b_dir = f'{out_dir}/train_B'
    output_a_mask_dir = f'{out_dir}/train_maskA'
    output_b_mask_dir = f'{out_dir}/train_maskB'

    output_a_val_dir = f'{out_dir}/val_A'
    output_b_val_dir = f'{out_dir}/val_B'
    output_a_mask_val_dir = f'{out_dir}/val_maskA'
    output_b_mask_val_dir = f'{out_dir}/val_maskB'

    output_a_test_dir = f'{out_dir}/test_A'
    output_b_test_dir = f'{out_dir}/test_B'
    output_a_mask_test_dir = f'{out_dir}/test_maskA'
    output_b_mask_test_dir = f'{out_dir}/test_maskB'

    os.makedirs(output_a_dir, exist_ok=True)
    os.makedirs(output_b_dir, exist_ok=True)
    os.makedirs(output_a_mask_dir, exist_ok=True)
    os.makedirs(output_b_mask_dir, exist_ok=True)

    os.makedirs(output_a_val_dir, exist_ok=True)
    os.makedirs(output_b_val_dir, exist_ok=True)
    os.makedirs(output_a_mask_val_dir, exist_ok=True)
    os.makedirs(output_b_mask_val_dir, exist_ok=True)

    os.makedirs(output_a_test_dir, exist_ok=True)
    os.makedirs(output_b_test_dir, exist_ok=True)
    os.makedirs(output_a_mask_test_dir, exist_ok=True)
    os.makedirs(output_b_mask_test_dir, exist_ok=True)

    ## Partition scan into 80/10/10 for train/val/test
    a_files = sorted(glob.glob(root_a))
    b_files = sorted(glob.glob(root_b))
    train_len = int(len(a_files)*0.8)
    val_len = int(len(a_files)*0.1)
    train_idx = np.arange(0, train_len)
    val_idx = np.arange(train_len, train_len + val_len)


    ## Resample resolutions as needed

    results = 'vis' # Visualize 2D slices for debugging purpose
    os.makedirs(results, exist_ok=True)

    crop = 0.0 # Crop first dimension, ignore some air only scans
    crop_h = 0.9 # Crop height dimension (dim 2)

    ## CT preprocess
    for idx, filepath in enumerate(tqdm(b_files)):
        img = ants.image_read(filepath)
        img = ants.resample_image(img, resample, False, 1)
        img = img.numpy()
        filename = os.path.splitext(os.path.basename(filepath))[0]
        img, mask = get_3d_mask(img, min_=min_ct, max_=max_ct, th=th)
        # Our scans have irregular size, crop to adjust, comment out as needed
        img, mask = crop_scan(img, mask, crop, ignore_zero=(idx in train_idx))
        if idx in train_idx:
            output_ct_dir = output_b_dir
            output_ct_mask_dir = output_b_mask_dir
        elif idx in val_idx:
            output_ct_dir = output_b_val_dir
            output_ct_mask_dir = output_b_mask_val_dir
        else:
            output_ct_dir = output_b_test_dir
            output_ct_mask_dir = output_b_mask_test_dir
        save_slice(img, mask, output_ct_dir, output_ct_mask_dir, filename)
        visualize(img, f'{results}/ct')
        visualize(mask, f'{results}/ct_mask')

    ## MRI preprocess
    for idx, filepath in enumerate(tqdm(a_files)):
        img = ants.image_read(filepath)
        img = ants.resample_image(img, resample, False, 1)
        img = img.numpy()
        filename = os.path.splitext(os.path.basename(filepath))[0]
        img, mask = get_3d_mask(img, min_=0, th=10, width=10)
        # Our scans have irregular size, crop to adjust, comment out as needed
        img, mask = crop_scan(img, mask, crop, ignore_zero=(idx in train_idx))
        if idx in train_idx:
            output_mri_dir = output_a_dir
            output_mri_mask_dir = output_a_mask_dir
        elif idx in val_idx:
            output_mri_dir = output_a_val_dir
            output_mri_mask_dir = output_a_mask_val_dir
        else:
            output_mri_dir = output_a_test_dir
            output_mri_mask_dir = output_a_mask_test_dir
        save_slice(img, mask, output_mri_dir, output_mri_mask_dir, filename)
        
        visualize(img, f'{results}/mri')
        visualize(mask, f'{results}/mri_mask')









