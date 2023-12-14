#CUDA_VISIBLE_DEVICES=0 python train.py --dataroot /data/paediatric-head/processed_img_v2 --gpu_ids 0 --display_id 0 --name cycle-corr --batchSize 64 --fineSize 160 --loadSize 198
#python train.py --dataroot /data/paediatric-head/processed_img_crop --gpu_ids 1 --display_id 0 --name cycle-structure-amp-crop --batchSize 42 --lambda_co_A 0 --lambda_co_B 0 --lambda_s_A 0.5 --lambda_s_B 0.5 --fineSize 200 --loadSize 224 --niter 50 --niter_decay 50 --serial_batches
#python train.py --dataroot /data/paediatric-head/processed_img_crop --gpu_ids 1 --display_id 0 --name unet-amp-crop --batchSize 128 --lambda_co_A 0 --lambda_co_B 0 --lambda_s_A 0.0 --lambda_s_B 0.0 --fineSize 256 --loadSize 280 --niter 50 --niter_decay 50 --serial_batches --which_model_netG unet_256

# hieu-torch attention-mask-shape
# python train.py --dataroot /data/data/paediatric-head/processed_img_open --gpu_ids 1 --display_id 0 --model mask_gan --name mask_gan --netG att \
# --dataset_mode unaligned --pool_size 50 --no_dropout --amp_level O0 \
# --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --lambda_mask 1.0 --lambda_shape 0.5 --load_size 150 --pad_size 225 --crop_size 224 --preprocess resize_pad_crop --no_flip \
# --batch_size 4 --niter 40 --niter_decay 40 --display_freq 1000 --print_freq 1000 --n_attentions 5

python train.py --dataroot /data/data/processed-pelvis-mr-ct-mask --gpu_ids 0 --display_id 0 --model mask_gan --name hieu_mr_ct_pelvis_attgan --dataset_mode unaligned --pool_size 50 --no_dropout \
--norm instance --lambda_mask 0.0 --lambda_co_B 0.0 --preprocess none --opt_level O1 --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --lambda_shape 0.0 \
--batch_size 12 --niter 50 --niter_decay 50 --display_freq 1000 --print_freq 1000 --Aclass A --Bclass B --half
