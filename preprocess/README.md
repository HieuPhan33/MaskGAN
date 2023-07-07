# Preprocess MR-CT data and generate masks
- For simplicity, we assume the dataset have all pairs MRI-CT.
- The simplified code only has a single for-loop to partition 80/20/20 train/val/test. 
- If you have an unpaired training set, i.e., source and target modalities do not match. You can simply separate the training data preprocessing and val/test data preprocessing by copy-paste the for-loop.

## Environment installation
Setup using `pip install -r requirements.txt`

## Data preparation
- Refer to your root folder as `root`. Assume your data structure is as follows
```bash
├── root/
│   ├── MRI/
│   │   ├── filename001.nii
│   │   ├── filename002.nii
│   │   └── ...
│   └── CT/
│       ├── filename001.nii
│       ├── filename002.nii
│       └── ...
```
- If your data structure is different, please modify the pattern matching expression at lines 138-139 in `preprocess/main.py`:
```python
root_a = f'{data_dir}/MRI/*.nii'
root_b = f'{data_dir}/CT/*.nii'
```

## Preprocess
3 arguments that you can specify:
- `--root`: data root;
- `--out`: output directory name, default is processed-mr-ct;
- `--resample`: resample the resolution of the medical scans, default is [1.0, 1.0, 1.0] mm^3.

Since our paediatric scans have irregular sizes, we need to crop the depth and height dimensions in function `crop_scan()` at Ln 47. When running, the preprocessed 2D slice visualizations are saved under `vis` for your inspection. Use them to modify data augmentation `crop_scan()` as needed.
