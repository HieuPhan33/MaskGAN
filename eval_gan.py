# from frechet_kernel_Inception_distance import *
# from inception_score import *
from glob import glob
import os
import argparse
from skimage.metrics import structural_similarity
import math
import imageio, cv2
import numpy as np

def get_images(filename, channels=1, size=None):
    # x = misc.imread(filename)
    # x = misc.imresize(x, size=[224, 224])
    x = imageio.imread(filename)
    if size is not None:
        x = cv2.resize(x, (size,size))
    if channels == 3:
        x = np.stack((x,)*3, axis=-1)
    return x

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def error(target_dir, fake_dir) :
    filenames = glob(os.path.join(target_dir, '*.*'))
    real_images = [get_images(filename) for filename in filenames]
    #real_images = np.transpose(real_images,axes=[0, 3, 1, 2])

    #filenames = glob(os.path.join(fake_dir, '*.*'))
    filenames = [s.replace('real', 'fake') for s in filenames]
    fake_images = [get_images(filename) for filename in filenames]
    #fake_images = np.transpose(fake_images,axes=[0, 3, 1, 2])
    ssim = np.mean([structural_similarity(real, fake) for real, fake in zip(real_images, fake_images)])
    psnr = np.mean([PSNR(real, fake) for real, fake in zip(real_images, fake_images)])
    fake_images = np.array(fake_images)
    real_images = np.array(real_images)
    mae = np.mean(np.absolute((real_images.astype("float") - fake_images.astype("float"))))
    return ssim, mae, psnr



def inception_score(fake_dir) :
    filenames = glob(os.path.join(fake_dir, '*.*'))
    images = [get_images(filename, channels=3, size=299) for filename in filenames]
    images = np.transpose(images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])

    logits = inception_logits(inception_images)

    IS = get_inception_score(BATCH_SIZE, images, inception_images, logits, splits=10)
    return IS
    


def frechet_inception_distance(target_dir, fake_dir) :
    filenames = glob(os.path.join(target_dir, '*.*'))
    real_images = [get_images(filename, channels=3, size=299) for filename in filenames]
    real_images = np.transpose(real_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join(fake_dir, '*.*'))
    fake_images = [get_images(filename, channels=3, size=299) for filename in filenames]
    fake_images = np.transpose(fake_images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

    fcd = frechet_classifier_distance_from_activations(real_activation, fake_activation)
    activations = inception_activations(inception_images)

    FID = get_fid(fcd, BATCH_SIZE, real_images, fake_images, inception_images, real_activation, fake_activation, activations)
    return FID


def kernel_inception_distance(target_dir, fake_dir) :
    filenames = glob(os.path.join(target_dir, '*.*'))
    real_images = [get_images(filename, channels=3, size=299) for filename in filenames]
    real_images = np.transpose(real_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join(fake_dir, '*.*'))
    fake_images = [get_images(filename, channels=3, size=299) for filename in filenames]
    fake_images = np.transpose(fake_images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

    kcd_mean, kcd_stddev = kernel_classifier_distance_and_std_from_activations(real_activation, fake_activation, max_block_size=10)
    activations = inception_activations(inception_images)

    KID_mean = get_kid(kcd_mean, BATCH_SIZE, real_images, fake_images, inception_images, real_activation, fake_activation, activations)
    KID_stddev = get_kid(kcd_stddev, BATCH_SIZE, real_images, fake_images, inception_images, real_activation, fake_activation, activations)
    return KID_mean, KID_stddev

def mean_kernel_inception_distance() :
    source_alpha = 0.98
    target_alpha = 1 - source_alpha
    
    filenames = glob(os.path.join('./real_source', '*.*'))
    real_source_images = [get_images(filename) for filename in filenames]
    real_source_images = np.transpose(real_source_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join('./real_target', '*.*'))
    real_target_images = [get_images(filename) for filename in filenames]
    real_target_images = np.transpose(real_target_images, axes=[0, 3, 1, 2])

    filenames = glob(os.path.join('./fake', '*.*'))
    fake_images = [get_images(filename) for filename in filenames]
    fake_images = np.transpose(fake_images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 1

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    real_activation = tf.placeholder(tf.float32, [None, None], name='activations1')
    fake_activation = tf.placeholder(tf.float32, [None, None], name='activations2')

    fcd = frechet_classifier_distance_from_activations(real_activation, fake_activation)
    kcd_mean, kcd_stddev = kernel_classifier_distance_and_std_from_activations(real_activation, fake_activation,
                                                                               max_block_size=10)
    activations = inception_activations(inception_images)

    FID = get_fid(fcd, BATCH_SIZE, real_target_images, fake_images, inception_images, real_activation, fake_activation, activations)
    KID_mean = get_kid(kcd_mean, BATCH_SIZE, real_target_images, fake_images, inception_images, real_activation, fake_activation, activations)
    KID_stddev = get_kid(kcd_stddev, BATCH_SIZE, real_target_images, fake_images, inception_images, real_activation, fake_activation, activations)

    mean_FID = get_fid(fcd, BATCH_SIZE, real_source_images, fake_images, inception_images, real_activation, fake_activation, activations)
    mean_KID_mean = get_kid(kcd_mean, BATCH_SIZE, real_source_images, fake_images, inception_images, real_activation, fake_activation, activations)
    mean_KID_stddev = get_kid(kcd_stddev, BATCH_SIZE, real_source_images, fake_images, inception_images, real_activation, fake_activation, activations)

    mean_FID = (target_alpha * FID + source_alpha * mean_FID) / 2.0
    mean_KID_mean = (target_alpha * KID_mean + source_alpha * mean_KID_mean) / 2.0
    mean_KID_stddev = (target_alpha * KID_stddev + source_alpha * mean_KID_stddev) / 2.0

    # mean_FID = (2 * FID * mean_FID) / (FID + mean_FID)
    # mean_KID_mean = (2 * KID_mean * mean_KID_mean) / (KID_mean + mean_KID_mean)
    # mean_KID_stddev = (2 * KID_stddev * mean_KID_stddev) / (KID_stddev + mean_KID_stddev)
    print("mean_FID : ", mean_FID / 100)
    print("mean_KID_mean : ", mean_KID_mean * 100)
    print("mean_KID_stddev : ", mean_KID_stddev * 100)



parser = argparse.ArgumentParser()
parser.add_argument('--fake-dir', type=str, help='fake directory')
parser.add_argument('--target-dir', type=str, help='target directory')
opt = parser.parse_args()
fake_dir = opt.fake_dir
target_dir = opt.target_dir
ssim, mae, psnr = error(fake_dir=fake_dir, target_dir=target_dir)
# IS = inception_score(fake_dir)
# FID = frechet_inception_distance(fake_dir=fake_dir, target_dir=target_dir)
# KID_mean, KID_stddev = kernel_inception_distance(fake_dir=fake_dir, target_dir=target_dir)
# mean_kernel_inception_distance()

print("==========")
print("SSIM ", ssim)
print("MAE ", mae)
print("PSNR", psnr)
# print("IS : ", IS)
# print("FID : ", FID / 100)
# print("KID_mean : ", KID_mean * 100)
# print("KID_stddev : ", KID_stddev * 100)

