# Preprocess MR-CT data and generate masks
- For simplicity, we assume the dataset have MRI and CT scans.
- This code assumes that you have the raw data in three folders: **train**, **val** and **test**. 
- **train** set contains only <ins>unpaired images</ins>, whereas **val** and **test** set contain <ins>paired images</ins>. 

## Environment installation
Setup using `pip install -r requirements.txt`

## Data preparation
- Refer to your root folder as `root`. Assume your data structure is as follows
```bash
├── root/
   ├── train/
   |   ├── MRI/
   │   │   ├── filename001.nii
   │   │   ├── filename002.nii
   │   │   └── ...
   |   └── CT/
   |       ├── filename001.nii
   |       ├── filename002.nii
   |       └── ...   
   ├── val/
   |   ├── MRI/
   |   |   ├── filename003.nii
   |   |   └── filename004.nii
   |   └── CT/
   |       ├── filename003.nii
   |       └── filename004.nii
   └── test/
       ├── MRI/
       |   ├── filename005.nii
       |   └── filename006.nii
       └── CT/
           ├── filename005.nii
           └── filename006.nii

```

## Preprocess
3 arguments that you can specify:
- `--root`: data root;
- `--out`: output directory name, default is processed-mr-ct;
- `--resample`: resample the resolution of the medical scans, default is [1.0, 1.0, 1.0] mm^3.

Since our paediatric scans have irregular sizes, we need to crop the depth and height dimensions in function `crop_scan()` at Ln 47. When running, the preprocessed 2D slice visualizations are saved under `vis` for your inspection. Use them to modify data augmentation `crop_scan()` as needed.

## Update!

Now, to use the based position selection strategy, our preprocessing stage generate files as: filename_XXX.jpg, where XXX corresponds to the relative position of the slice respect to the entire volumetric image. In that way, our model can choose slices of similar position.
