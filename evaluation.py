import numpy as np 
import os
import argparse
import os
from skimage.metrics import structural_similarity
from math import log10, sqrt 
import argparse
import matplotlib.image as mpimg

# Call the parser
parser = argparse.ArgumentParser()
parser.add_argument('--results_folder', default='testing/', help='path to results')
args = parser.parse_args()

# Definition of folders
real_B_path = 'results/' + args.results_folder  +'real_B/'
real_A_path = 'results/' + args.results_folder + 'real_A/'
fake_B_path = 'results/' + args.results_folder  +'fake_B/'
fake_A_path = 'results/' + args.results_folder + 'fake_A/'

# Get list of file
real_imagesB = os.listdir(real_B_path)
real_imagesB = sorted(real_imagesB)
real_imagesA = os.listdir(real_A_path)
real_imagesA = sorted(real_imagesA)
fake_imagesB = os.listdir(fake_B_path)
fake_imagesB = sorted(fake_imagesB)
fake_imagesA = os.listdir(fake_A_path)
fake_imagesA = sorted(fake_imagesA)

# Definition of peak signal noise ratio
def psnr_function(img_a, img_b, modality):

    mse = np.mean((img_a - img_b) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                # Therefore PSNR have no importance. 
        return 100.0
    if modality == "mri2ct":
        psnr_val = 20 * log10(2000 / sqrt(mse)) 
    elif modality == "ct2mri":
        psnr_val = 20 * log10(2000 / sqrt(mse)) 

    return psnr_val 

def ssim_function(img_a, img_b, modality):
    if modality == 'mri2ct':
        # We add 800 to avoid negative values
        img_a = ((img_a) + 800)
        img_b = ((img_b) + 800)
        return structural_similarity(img_a, img_b, data_range = 2800) 
    elif modality == 'ct2mri':
        return structural_similarity(img_a, img_b, data_range = 2000) 

# Definition of metrics
def getting_metrics(img_a, img_b, modality):
    # Getting MAE
    metric_mae = np.absolute(np.subtract(img_a, img_b)).mean()
    # Getting SSIM. It is necessary to set a distance between max and min value
    metric_ssim = ssim_function(img_a, img_b, modality)
    # Getting PSNR
    metric_psnr = psnr_function(img_a,img_b, modality)
    return metric_mae, metric_ssim, metric_psnr

# Set the metrics to zero for the translation from MRI to CT   
mae_mri2ct = 0
ssim_mri2ct = 0
psnr_mri2ct = 0
# Set the metrics to zero for the translation from CT to MRI  
mae_ct2mri = 0
ssim_ct2mri = 0
psnr_ct2mri = 0

# Do a loop for every predicted image
for img in range(len(real_imagesB)):

    # Read each image (real and fake)
    real_imgB = mpimg.imread(f"{real_B_path}{real_imagesB[img]}")
    real_imgA = mpimg.imread(f"{real_A_path}{real_imagesA[img]}")
    fake_imgB = mpimg.imread(f"{fake_B_path}{fake_imagesB[img]}")
    fake_imgA = mpimg.imread(f"{fake_A_path}{fake_imagesA[img]}")

    # Modyfing the values of images to get original ranges
    real_imgB = ((real_imgB* (2000 - (-800)))) + (-800)
    real_imgB = np.clip(real_imgB,a_min= -800, a_max= 2000)
    real_imgA = ((real_imgA* (2000 - (0)))) + (0)
    real_imgA = np.clip(real_imgA,a_min= 0, a_max= 2000)
    fake_imgA = ((fake_imgA* (2000 - (0)))) + (0)
    fake_imgA = np.clip(fake_imgA,a_min= 0, a_max= 2000)
    fake_imgB = ((fake_imgB* (2000 - (-800)))) + (-800)
    fake_imgB = np.clip(fake_imgB,a_min= -800, a_max= 2000)

    # Compute metrics for the current slice
    mae_ct, ssim_ct, psnr_ct = getting_metrics(real_imgB,fake_imgB, modality = "mri2ct")
    mae_mri, ssim_mri, psnr_mri = getting_metrics(real_imgA,fake_imgA, modality = "ct2mri")

    # Accumulate metrics for generation of synthetic CT
    mae_mri2ct += mae_ct
    ssim_mri2ct += ssim_ct
    psnr_mri2ct += psnr_ct
    # Accumulate metrics for generation of synthetic MRI
    mae_ct2mri += mae_mri
    ssim_ct2mri+= ssim_mri
    psnr_ct2mri += psnr_mri

# Getting the averaged metrics
print("For the generation of synthetic CT, results are:")
print("================================================")
print(f"MAE: {mae_mri2ct/len(real_imagesB)}")
print(f"SSIM: {ssim_mri2ct/len(real_imagesB)}")
print(f"PSNR: {psnr_mri2ct/len(real_imagesB)}")
print("")
print("For the generation of synthetic CT, results are:")
print("================================================")
print(f"MAE: {mae_ct2mri/len(real_imagesB)}")
print(f"SSIM: {ssim_ct2mri/len(real_imagesB)}")
print(f"PSNR: {psnr_ct2mri/len(real_imagesB)}")


