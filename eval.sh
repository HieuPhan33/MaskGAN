# python main.py --fake-dir /media/hieu/DATA/medical/01-code/Unpaired_MR_to_CT_Image_Synthesis/results/corr_gan2/fake_CT \
# --target-dir /media/hieu/DATA/medical/01-code/Unpaired_MR_to_CT_Image_Synthesis/results/corr_gan2/real_CT

# exp_name=resvit
# python main.py --fake-dir /data/01-code/cycle-transformer/results/${exp_name}/fake_MRI \
# --target-dir /data/01-code/cycle-transformer/results/${exp_name}/real_MRI

exp_name=exp_name
dir='results'

echo "===== ${exp_name} ======="
python main.py --fake-dir ${dir}/${exp_name}/fake_A \
--target-dir ${dir}/${exp_name}/real_A

python main.py --fake-dir ${dir}/${exp_name}/fake_B \
--target-dir ${dir}/${exp_name}/real_B
