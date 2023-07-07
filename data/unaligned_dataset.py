import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
from torchvision import io

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + f'_{opt.Aclass}')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + f'_{opt.Bclass}')  # create a path '/path/to/data/trainB'
        self.dir_maskA = os.path.join(opt.dataroot, opt.phase + f'_mask{opt.Aclass}')  # create a path '/path/to/data/trainA'
        self.dir_maskB = os.path.join(opt.dataroot, opt.phase + f'_mask{opt.Bclass}')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        self.maskA_paths = sorted(make_dataset(self.dir_maskA, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.maskB_paths = sorted(make_dataset(self.dir_maskB, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = opt.direction == 'BtoA'
        if opt.half:
            self.A_size, self.B_size = self.A_size//2, self.B_size//2
            self.A_paths, self.B_paths = self.A_paths[:self.A_size], self.B_paths[self.B_size:]
        
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(self.input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(self.output_nc == 1))
        # self.transform_maskA = get_transform(self.opt, grayscale=(self.input_nc == 1), mask=True)
        # self.transform_maskB = get_transform(self.opt, grayscale=(self.output_nc == 1), mask=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        maskA_path = self.maskA_paths[index_A]
        maskB_path = self.maskB_paths[index_B]
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')

        # A_mask = Image.open(maskA_path)
        # B_mask = Image.open(maskB_path)
        A_img = io.read_image(A_path)
        B_img = io.read_image(B_path)

        A_mask = io.read_image(maskA_path)
        B_mask = io.read_image(maskB_path)
        
        # apply image transformation
        A, A_mask = self.transform_A(A_img, A_mask)
        B, B_mask = self.transform_B(B_img, B_mask)

        # A_mask = self.transform_maskA(A_mask)
        # B_mask = self.transform_maskB(B_mask)

        # if self.input_nc == 1:  # RGB to gray
        #     tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        #     A = tmp.unsqueeze(0)

        # if self.output_nc == 1:  # RGB to gray
        #     tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
        #     B = tmp.unsqueeze(0)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path,
        'A_mask': A_mask, 'B_mask': B_mask}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
