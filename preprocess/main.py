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
from scipy import ndimage
from skimage.morphology import binary_erosion, binary_dilation

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
    img = np.transpose(img, (0,2,1))[:,::-1,::-1]
    if mask is not None:
        mask = np.transpose(mask, (0,2,1))[:,::-1,::-1]
    if ignore_zero:
        mask_ = img.sum(axis=(1,2)) > 0
        img = img[mask_]
        if mask is not None:
            mask = mask[mask_]
    if crop > 0:
        length = img.shape[0]
        img = img[int(crop*length): int((1-crop)*length)]
        if mask is not None:
            mask = mask[int(crop*length): int((1-crop)*length)]
    
    if crop_h > 0:
        if img.shape[1] > 200:
            crop_h = 0.8
        new_h = int(crop_h*img.shape[1])
        img = img[:, :new_h]
        if mask is not None:
            mask = mask[:, :new_h]

    return img, mask


def crop_scan_paired(img1, img2, mask, crop=0, crop_h=0, ignore_zero=True):
    img1 = np.transpose(img1, (0,2,1))[:,::-1,::-1]
    img2 = np.transpose(img2, (0,2,1))[:,::-1,::-1]
    if mask is not None:
        mask = np.transpose(mask, (0,2,1))[:,::-1,::-1]
    if ignore_zero:
        mask1_ = img1.sum(axis=(1,2)) > 0
        mask2_ = img2.sum(axis=(1,2)) > 0
        mask_ = mask1_ * mask2_
        img1 = img1[mask_]
        img2 = img2[mask_]
        if mask is not None:
            mask = mask[mask_]
    if crop > 0:
        length = img1.shape[0]
        img1 = img1[int(crop*length): int((1-crop)*length)]
        img2 = img2[int(crop*length): int((1-crop)*length)]
        if mask is not None:
            mask = mask[int(crop*length): int((1-crop)*length)]
    
    if crop_h > 0:
        if img1.shape[1] > 200:
            crop_h = 0.8
        new_h = int(crop_h*img1.shape[1])
        img1 = img1[:, :new_h]
        img2 = img2[:, :new_h]
        if mask is not None:
            mask = mask[:, :new_h]

    return img1, img2, mask

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

    ## Remove holes
    mask = np.zeros(img.shape).astype(int)
    mask[img > th] = 1

    ## Opening np.ones((3,3,3))
    mask = morphology.binary_opening(mask, )

    remove_holes = morphology.remove_small_holes(
        mask, 
        area_threshold=width ** 3
    )

    largest_cc = getLargestCC(remove_holes)
    return img, largest_cc.astype(int)

def float_to_padded_string(number, total_digits=3):
    formatted_number = format(number, f".{total_digits}f")
    return formatted_number.lstrip('0.') or '0'

def resize_volume(img,desired_depth,desired_width, desired_height):

    current_depth = img.shape[0]
    current_width = img.shape[1]
    current_height = img.shape[2]
 
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    
    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
    return img


def save_slice(img, mask, data_dir, data_mask_dir, filename):
    assert img.shape == mask.shape, f"Shape not match - img {img.shape} vs mask {mask.shape}"
    pad_width = ((5, 5), (5, 5), (5, 5)) 
    # Resize to 204,204,204
    img = resize_volume(img, 214,214,214)
    img = np.pad(img, pad_width, mode='constant', constant_values=0)
    mask = resize_volume(mask, 214,214,214)
    mask = np.pad(mask, pad_width, mode='constant', constant_values=0)
    
    for i in range(len(img)):
        #im = np.uint8(255*normalize(img[i]))
        im = img[i]
        m = 255*mask[i].astype(np.uint8)
        
        #m = np.uint8(255*normalize(mask[i]))
        imageio.imwrite(f'{data_dir}/{filename}_{str(i).zfill(3)}_{float_to_padded_string(round(i/len(img),2), 3)}.png', im)
        imageio.imwrite(f'{data_mask_dir}/{filename}_{str(i).zfill(3)}_{float_to_padded_string(round(i/len(img),2), 3)}.png', m)


########################################################
###           DEFINITION OF DIRECTORIES              ###
########################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data')
    # Add arguments
    parser.add_argument('--root', type=str, help='Data root dir')
    parser.add_argument('--out', type=str, default='processed-mr-ct', help='Output directory')
    parser.add_argument('--resample', nargs='+', type=float, default=[1.0, 1.0, 1.0], help='Resample scan resolutions')
    args = parser.parse_args()
    # Declaring variables
    data_dir = args.root
    out_dir = args.out
    resample = args.resample

    # Defining root folders (nii files)
    train_root =  f'{data_dir}/train/'
    val_root =  f'{data_dir}/val/'
    test_root =  f'{data_dir}/test/'

    # Defining output folders
    train_ct = train_root + 'ct/*.nii.gz'
    train_mri = train_root + 'mri/*.nii.gz'
    val_ct = val_root + 'ct/*.nii.gz'
    val_mri = val_root + 'mri/*.nii.gz'
    test_ct = test_root + 'ct/*.nii.gz'
    test_mri = test_root + 'mri/*.nii.gz'

    output_ct_dir = f'{out_dir}/train_B'
    output_mri_dir = f'{out_dir}/train_A'
    output_ct_mask_dir = f'{out_dir}/train_maskB'
    output_mri_mask_dir = f'{out_dir}/train_maskA'

    # Create main directories
    os.makedirs(output_ct_dir, exist_ok=True)
    os.makedirs(output_mri_dir, exist_ok=True)
    os.makedirs(output_ct_mask_dir, exist_ok=True)
    os.makedirs(output_mri_mask_dir, exist_ok=True)

    ct_files_train = glob.glob(train_ct)
    mri_files_train = glob.glob(train_mri)
    #min_ct, max_ct = -800, 1900
    th_ct = 50
    th_mri = 10
    # Set clip CT intensity
    min_ct, max_ct = -800, 2000
    #th = 10 # Consider pixel less than certain threshold as background (remove noise, artifacts)

    results = 'vis'
    os.makedirs(results, exist_ok=True)

    crop = 0.0
    crop_h = 0.9
    resample = [1.0, 1.0, 1.0]

    print("Creating MR images for training")
    for idx, filepath in enumerate(mri_files_train):

        mri = ants.image_read(filepath)
        mri = ants.resample_image(mri, resample, False, 1).numpy()
        #filename = os.path.splitext(os.path.basename(filepath))[0]
        mri, mask = get_3d_mask(mri, min_=0, th=th_mri, width=10)
        # Our scans have irregular size, crop to adjust, comment out as needed
        mri, mask = crop_scan(mri, mask, crop,crop_h)

        mri = mri.astype('uint8')
        mask = mask.astype('uint8')

        # Remove noise
        # Define the structure element for erosion
        selem_ero = np.ones((1, 1, 1), dtype=bool) 
        selem_dil = np.ones((10, 10, 10), dtype=bool) 
        # Perform erosion on the entire 3D array
        eroded_mask = binary_erosion(mask, selem_ero)
        mask[eroded_mask == 0] = 0
        dilated_mask = binary_dilation(mask, selem_dil)
        mri[dilated_mask == 0] = 0
        mask[dilated_mask== 0] = 0

        # Remove images with zero values in the mask
        non_zero_slices_mask_axis1_2 = np.any(mask, axis=(1, 2))
        mri = mri[non_zero_slices_mask_axis1_2]
        mask = mask[non_zero_slices_mask_axis1_2]
        non_zero_slices_mask_axis0_1 = np.any(mask, axis=(0, 1))
        mri = mri[:,:,non_zero_slices_mask_axis0_1]
        mask = mask[:,:,non_zero_slices_mask_axis0_1]
        non_zero_slices_mask_axis0_2 = np.any(mask, axis=(0, 2))
        mri = mri[:,non_zero_slices_mask_axis0_2,:]
        mask = mask[:,non_zero_slices_mask_axis0_2,:]

        # Enter the name of the file
        filename = filepath.split('/')[-1].replace('.nii.gz','')
        # Create a generic format
        filename = filename.zfill(3)
        save_slice(mri, mask, output_mri_dir, output_mri_mask_dir, filename)
    
    print("Creating CT images for training")
    for idx, filepath in enumerate(ct_files_train): 

        ct = ants.image_read(filepath)
        ct = ants.resample_image(ct, resample, False, 1).numpy()
        ct, mask = get_3d_mask(ct, min_=min_ct, max_=max_ct, th=th_ct)
        # Our scans have irregular size, crop to adjust, comment out as needed
        ct, mask = crop_scan(ct, mask, crop,crop_h)

        ct = ct.astype('uint8')
        mask = mask.astype('uint8')

        # Remove noise
        # Define the structure element for erosion
        selem_ero = np.ones((1, 1, 1), dtype=bool) 
        selem_dil = np.ones((10, 10, 10), dtype=bool) 
        # Perform erosion on the entire 3D array
        eroded_mask = binary_erosion(mask, selem_ero)
        mask[eroded_mask == 0] = 0
        dilated_mask = binary_dilation(mask, selem_dil)
        ct[dilated_mask == 0] = 0
        mask[dilated_mask== 0] = 0

        # Remove images with zero values in the mask
        non_zero_slices_mask_axis1_2 = np.any(mask, axis=(1, 2))
        ct = ct[non_zero_slices_mask_axis1_2]
        mask = mask[non_zero_slices_mask_axis1_2]

        non_zero_slices_mask_axis0_1 = np.any(mask, axis=(0, 1))
        ct = ct[:,:,non_zero_slices_mask_axis0_1]
        mask = mask[:,:,non_zero_slices_mask_axis0_1]

        non_zero_slices_mask_axis0_2 = np.any(mask, axis=(0, 2))
        ct = ct[:,non_zero_slices_mask_axis0_2,:]
        mask = mask[:,non_zero_slices_mask_axis0_2,:]

        # Enter the name of the file
        filename = filepath.split('/')[-1].replace('.nii.gz','')
        # Create a generic format
        filename = filename.zfill(3)

        save_slice(ct, mask, output_ct_dir, output_ct_mask_dir, filename)

    ### VALIDATION
    output_ct_dir = f'{out_dir}/val_B'
    output_mri_dir = f'{out_dir}/val_A'
    output_ct_mask_dir = f'{out_dir}/val_maskB'
    output_mri_mask_dir = f'{out_dir}/val_maskA'

    os.makedirs(output_ct_dir, exist_ok=True)
    os.makedirs(output_mri_dir, exist_ok=True)
    os.makedirs(output_ct_mask_dir, exist_ok=True)
    os.makedirs(output_mri_mask_dir, exist_ok=True)

    ct_files_val = glob.glob(val_ct)
    mri_files_val = glob.glob(val_mri)

    print("Creating MRI images and CT scans for validation")
    for mri_path, ct_path in zip(mri_files_val,ct_files_val):

        mri = ants.image_read(mri_path)
        mri = ants.resample_image(mri, resample, False, 1).numpy()
        #filename = os.path.splitext(os.path.basename(filepath))[0]
        mri, mri_mask = get_3d_mask(mri, min_=0, th=th_mri, width=10)

        ct = ants.image_read(ct_path)
        ct = ants.resample_image(ct, resample, False, 1).numpy()
        ct, ct_mask = get_3d_mask(ct, min_=min_ct, max_=max_ct, th=th_ct)

        # Getting a uniform mask template for paired images
        uniform_mask = mri_mask * ct_mask

        ct, mri, uniform_mask = crop_scan_paired(ct, mri, uniform_mask,crop,crop_h)
        mri_mask = uniform_mask
        ct_mask = uniform_mask
        
        ct = ct.astype('uint8')
        ct_mask = ct_mask.astype('uint8')
        mri = mri.astype('uint8')
        mri_mask = mri_mask.astype('uint8')

        # Remove noise
        selem_ero = np.ones((1, 1, 1), dtype=bool) 
        selem_dil = np.ones((10, 10, 10), dtype=bool) 
        # Perform erosion on the entire 3D array
        eroded_mask = binary_erosion(ct_mask, selem_ero)
        ct_mask[eroded_mask == 0] = 0
        dilated_mask = binary_dilation(ct_mask, selem_dil)
        ct[dilated_mask == 0] = 0
        ct_mask[dilated_mask== 0] = 0
        # Perform erosion on the entire 3D array
        eroded_mask = binary_erosion(mri_mask, selem_ero)
        mri_mask[eroded_mask == 0] = 0
        dilated_mask = binary_dilation(mri_mask, selem_dil)
        mri[dilated_mask == 0] = 0
        mri_mask[dilated_mask== 0] = 0

        # Remove images with zero values in the mask
        non_zero_slices_mask_axis1_2 = np.any(ct_mask, axis=(1, 2))
        ct = ct[non_zero_slices_mask_axis1_2]
        ct_mask = ct_mask[non_zero_slices_mask_axis1_2]
        non_zero_slices_mask_axis0_1 = np.any(ct_mask, axis=(0, 1))
        ct = ct[:,:,non_zero_slices_mask_axis0_1]
        ct_mask = ct_mask[:,:,non_zero_slices_mask_axis0_1]
        non_zero_slices_mask_axis0_2 = np.any(ct_mask, axis=(0, 2))
        ct = ct[:,non_zero_slices_mask_axis0_2,:]
        ct_mask = ct_mask[:,non_zero_slices_mask_axis0_2,:]

        non_zero_slices_mask_axis1_2 = np.any(mri_mask, axis=(1, 2))
        mri = mri[non_zero_slices_mask_axis1_2]
        mri_mask = mri_mask[non_zero_slices_mask_axis1_2]
        non_zero_slices_mask_axis0_1 = np.any(mri_mask, axis=(0, 1))
        mri = mri[:,:,non_zero_slices_mask_axis0_1]
        mri_mask = mri_mask[:,:,non_zero_slices_mask_axis0_1]
        non_zero_slices_mask_axis0_2 = np.any(mri_mask, axis=(0, 2))
        mri = mri[:,non_zero_slices_mask_axis0_2,:]
        mri_mask = mri_mask[:,non_zero_slices_mask_axis0_2,:]

        # Enter the name of the file
        filename = mri_path.split('/')[-1].replace('.nii.gz','')
        # Create a generic format
        filename = filename.zfill(3)
        
        save_slice(mri, mri_mask, output_mri_dir, output_mri_mask_dir, filename)
        save_slice(ct, ct_mask, output_ct_dir, output_ct_mask_dir, filename)

    ### TESTING
    output_ct_dir = f'{out_dir}/test_B'
    output_mri_dir = f'{out_dir}/test_A'
    output_ct_mask_dir = f'{out_dir}/test_maskB'
    output_mri_mask_dir = f'{out_dir}/test_maskA'

    os.makedirs(output_ct_dir, exist_ok=True)
    os.makedirs(output_mri_dir, exist_ok=True)
    os.makedirs(output_ct_mask_dir, exist_ok=True)
    os.makedirs(output_mri_mask_dir, exist_ok=True)

    mri_files_test = glob.glob(test_mri)
    ct_files_test = glob.glob(test_ct)

    print("Creating MRI images and CT scans for testing")
    for mri_path, ct_path in zip(mri_files_test,ct_files_test):

        mri = ants.image_read(mri_path)
        mri = ants.resample_image(mri, resample, False, 1).numpy()
        #filename = os.path.splitext(os.path.basename(filepath))[0]
        mri, mri_mask = get_3d_mask(mri, min_=0, th=th_mri, width=10)

        ct = ants.image_read(ct_path)
        ct = ants.resample_image(ct, resample, False, 1).numpy()
        ct, ct_mask = get_3d_mask(ct, min_=min_ct, max_=max_ct, th=th_ct)

        # Getting a uniform mask template for paired images
        uniform_mask = mri_mask * ct_mask
        # Our scans have irregular size, crop to adjust, comment out as needed
        ct, mri, uniform_mask = crop_scan_paired(ct, mri, uniform_mask,crop,crop_h)
        mri_mask = uniform_mask
        ct_mask = uniform_mask

        ct = ct.astype('uint8')
        ct_mask = ct_mask.astype('uint8')
        mri = mri.astype('uint8')
        mri_mask = mri_mask.astype('uint8')

        # Remove noise
        selem_ero = np.ones((1, 1, 1), dtype=bool) 
        selem_dil = np.ones((10, 10, 10), dtype=bool) 
        # Perform erosion on the entire 3D array
        eroded_mask = binary_erosion(ct_mask, selem_ero)
        ct_mask[eroded_mask == 0] = 0
        dilated_mask = binary_dilation(ct_mask, selem_dil)
        ct[dilated_mask == 0] = 0
        ct_mask[dilated_mask== 0] = 0
        # Perform erosion on the entire 3D array
        eroded_mask = binary_erosion(mri_mask, selem_ero)
        mri_mask[eroded_mask == 0] = 0
        dilated_mask = binary_dilation(mri_mask, selem_dil)
        mri[dilated_mask == 0] = 0
        mri_mask[dilated_mask== 0] = 0

        # Remove images with zero values in the mask
        non_zero_slices_mask_axis1_2 = np.any(ct_mask, axis=(1, 2))
        ct = ct[non_zero_slices_mask_axis1_2]
        ct_mask = ct_mask[non_zero_slices_mask_axis1_2]
        non_zero_slices_mask_axis0_1 = np.any(ct_mask, axis=(0, 1))
        ct = ct[:,:,non_zero_slices_mask_axis0_1]
        ct_mask = ct_mask[:,:,non_zero_slices_mask_axis0_1]
        non_zero_slices_mask_axis0_2 = np.any(ct_mask, axis=(0, 2))
        ct = ct[:,non_zero_slices_mask_axis0_2,:]
        ct_mask = ct_mask[:,non_zero_slices_mask_axis0_2,:]

        non_zero_slices_mask_axis1_2 = np.any(mri_mask, axis=(1, 2))
        mri = mri[non_zero_slices_mask_axis1_2]
        mri_mask = mri_mask[non_zero_slices_mask_axis1_2]
        non_zero_slices_mask_axis0_1 = np.any(mri_mask, axis=(0, 1))
        mri = mri[:,:,non_zero_slices_mask_axis0_1]
        mri_mask = mri_mask[:,:,non_zero_slices_mask_axis0_1]
        non_zero_slices_mask_axis0_2 = np.any(mri_mask, axis=(0, 2))
        mri = mri[:,non_zero_slices_mask_axis0_2,:]
        mri_mask = mri_mask[:,non_zero_slices_mask_axis0_2,:]

        # Enter the name of the file
        filename = mri_path.split('/')[-1].replace('.nii.gz','')
        # Create a generic format
        filename = filename.zfill(3)
        
        save_slice(mri, mri_mask, output_mri_dir, output_mri_mask_dir, filename)
        save_slice(ct, ct_mask, output_ct_dir, output_ct_mask_dir, filename)


