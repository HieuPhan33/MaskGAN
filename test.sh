# python test.py --gpu_ids 0 --dataroot /data/data/paediatric-head/processed_img_open \
# mask_gan --netG att \
# --checkpoints_dir checkpoints --load_size 150 --pad_size 225 --crop_size 224 --preprocess resize_pad_crop \

python test.py --dataroot /data/data/processed-pelvis-mr-ct-mask --gpu_ids 0 --model mask_gan --name hieu_mr_ct_pelvis_attgan --dataset_mode unaligned --no_dropout \
--norm instance --preprocess none --Aclass A --Bclass B --netG att

