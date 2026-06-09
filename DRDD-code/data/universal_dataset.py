import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset, make_dataset_from_flist, make_dataset_all_text
from PIL import Image
import numpy as np
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import Augmentor
import cv2


class AlignedDataset_all(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.equalizeHist = equalizeHist
        self.augment_flip = augment_flip
        self.crop_patch = crop_patch
        self.generation = generation
        self.image_size = opt.image_size
        self.opt = opt

        # 新增：标记哪些任务使用随机配对
        
        if opt.phase == 'train':
            # LOL_all
            self.dir_Alol = os.path.join(opt.dataroot, 'LOL/train/low')
            self.dir_Blol = os.path.join(opt.dataroot, 'LOL/train/high')
            # Rain_all
            self.dir_Aderain = os.path.join(opt.dataroot, 'rain100H/train/rain')
            self.dir_Bderain = os.path.join(opt.dataroot, 'rain100H/train/norain')
            # self.dir_Aderain = os.path.join(opt.dataroot, 'syn_rain/train/Rain13K/input')
            # self.dir_Bderain = os.path.join(opt.dataroot, 'syn_rain/train/Rain13K/target')
            # Gopro
            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/train/input')
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/train/target')
            # desnow
            self.dir_Asnow = os.path.join(opt.dataroot, 'snow100k/input')
            self.dir_Bsnow = os.path.join(opt.dataroot, 'snow100k/gt')
            # dehazing
            # self.dir_Ahaze = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/hazy.flist')
            # self.dir_Bhaze = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/gt.flist')
            self.dir_Ahaze = os.path.join(opt.dataroot, 'dehaze/hazy.flist')
            self.dir_Bhaze = os.path.join(opt.dataroot, 'dehaze/clear.flist')

            self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train/input25')
            self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train/gt')

            self.dir_Ainpaint = '/root/data1/linziyue/RDDM/data/CelebA_HQ_mask/irregular'
            self.dir_Binpaint = '/root/data1/linziyue/RDDM/data/CelebA_HQ_256/train'

            self.dir_Atrans = '/root/data1/linziyue/RDDM/data/afhq/train/cat'
            self.dir_Btrans = '/root/data1/linziyue/RDDM/data/afhq/train/dog'

            if opt.dataset_type == 'data25':
                # LOL_all
                self.dir_Alol = os.path.join(opt.dataroot, 'LOL/train/low')
                self.dir_Blol = os.path.join(opt.dataroot, 'LOL/train/high')
                # Rain_all
                self.dir_Aderain = os.path.join(opt.dataroot, 'Rain100L/rain')
                self.dir_Bderain = os.path.join(opt.dataroot, 'Rain100L/rain')
                # Gopro
                self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/train/input')
                self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/train/target')
                # denoise
                self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train/input25')
                self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train/gt')
                # dehazing
                self.dir_Ahaze = os.path.join(opt.dataroot, 'dehaze/hazy_25.flist')
                self.dir_Bhaze = os.path.join(opt.dataroot, 'dehaze/clear_25.flist')
            # if opt.dataset_type == 'old':
            #     self.dir_Aderain = os.path.join(opt.dataroot, 'syn_rain/train/Rain13K/input')
            #     self.dir_Bderain = os.path.join(opt.dataroot, 'syn_rain/train/Rain13K/target')
            elif opt.dataset_type == 'new5':
                # LOL_all
                self.dir_Alol = os.path.join(opt.dataroot, 'LOL/train/low')
                self.dir_Blol = os.path.join(opt.dataroot, 'LOL/train/high')
                # self.dir_Alol = os.path.join(opt.dataroot, 'LOL/eval15/low')
                # self.dir_Blol = os.path.join(opt.dataroot, 'LOL/eval15/high')

                # Rain_all
                self.dir_Aderain = os.path.join(opt.dataroot, 'rain100H/train/rain')
                self.dir_Bderain = os.path.join(opt.dataroot, 'rain100H/train/norain')
                # Gopro
                self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/train/input')
                self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/train/target')
                # denoise
                # self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train/input25')
                # self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train/gt')

                self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train_new/input.flist')
                self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train_new/gt.flist')
                # dehazing
                # self.dir_Ahaze = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/hazy.flist')
                # self.dir_Bhaze = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/gt.flist')
                self.dir_Ahaze = '/root/data1/linziyue/RDDM/data/OTS/haze.flist'
                self.dir_Bhaze = '/root/data1/linziyue/RDDM/data/OTS/clean.flist'


            elif opt.dataset_type == 'new3_inpaint':
                # LOL_all
                # self.dir_Alol = os.path.join(opt.dataroot, 'LOL/train/low')
                # self.dir_Blol = os.path.join(opt.dataroot, 'LOL/train/high')
                # self.dir_Alol = os.path.join(opt.dataroot, 'LOL/eval15/low')
                # self.dir_Blol = os.path.join(opt.dataroot, 'LOL/eval15/high')
                # Rain_all
                self.dir_Aderain = os.path.join(opt.dataroot, 'Rain100L/rain')
                self.dir_Bderain = os.path.join(opt.dataroot, 'Rain100L/norain')
                # Gopro
                # denoise
                # self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train/input25')
                # self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train/gt')

                self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train_new/input.flist')
                self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train_new/gt.flist')
                # dehazing
                # self.dir_Ahaze = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/hazy.flist')
                # self.dir_Bhaze = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/gt.flist')
                self.dir_Ahaze = '/root/data1/linziyue/RDDM/data/OTS/haze.flist'
                self.dir_Bhaze = '/root/data1/linziyue/RDDM/data/OTS/clean.flist'

                self.dir_Ainpaint = '/root/data1/linziyue/RDDM/data/CelebA_HQ_mask/irregular'
                self.dir_Binpaint = '/root/data1/linziyue/RDDM/data/CelebA_HQ_256/train'

            elif opt.dataset_type == 'new5_inpaint':
                # LOL_all
                # self.dir_Alol = os.path.join(opt.dataroot, 'LOL/train/low')
                # self.dir_Blol = os.path.join(opt.dataroot, 'LOL/train/high')
                # self.dir_Alol = os.path.join(opt.dataroot, 'LOL/eval15/low')
                # self.dir_Blol = os.path.join(opt.dataroot, 'LOL/eval15/high')
                self.dir_Alol = os.path.join(opt.dataroot, 'LOL/our485/low')
                self.dir_Blol = os.path.join(opt.dataroot, 'LOL/our485/high')
                # Rain_all
                self.dir_Aderain = os.path.join(opt.dataroot, 'Rain100L/rain')
                self.dir_Bderain = os.path.join(opt.dataroot, 'Rain100L/norain')
                # Gopro
                self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/train/input')
                self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/train/target')
                # denoise
                # self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train/input25')
                # self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train/gt')

                self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train_new/input.flist')
                self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train_new/gt.flist')
                # dehazing
                # self.dir_Ahaze = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/hazy.flist')
                # self.dir_Bhaze = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/gt.flist')
                self.dir_Ahaze = '/root/data1/linziyue/RDDM/data/OTS/haze.flist'
                self.dir_Bhaze = '/root/data1/linziyue/RDDM/data/OTS/clean.flist'

                self.dir_Ainpaint = '/root/data1/houjiahe/Decouple_RDDM/DIV2K/DIV2K_train_LR_bicubic/X4_sub_480'
                self.dir_Binpaint = '/root/data1/houjiahe/Decouple_RDDM/DIV2K/DIV2K_train_HR_sub'

                self.dir_Atrans = '/root/data1/linziyue/RDDM/data/afhq/train/cat'
                self.dir_Btrans = '/root/data1/linziyue/RDDM/data/afhq/train/dog'
            # if opt.dataset_type == 'old':
            #     self.dir_Aderain = os.path.join(opt.dataroot, 'syn_rain/train/Rain13K/input')
            #     self.dir_Bderain = os.path.join(opt.dataroot, 'syn_rain/train/Rain13K/target')
            elif opt.dataset_type == 'rain25':
                self.dir_Aderain = os.path.join(opt.dataroot, 'rain100H/train25/rain')
                self.dir_Bderain = os.path.join(opt.dataroot, 'rain100H/train25/norain')
            elif opt.dataset_type == 'rain50':
                self.dir_Aderain = os.path.join(opt.dataroot, 'rain100H/train50/rain')
                self.dir_Bderain = os.path.join(opt.dataroot, 'rain100H/train50/norain')
            elif opt.dataset_type == 'rain75':
                self.dir_Aderain = os.path.join(opt.dataroot, 'rain100H/train75/rain')
                self.dir_Bderain = os.path.join(opt.dataroot, 'rain100H/train75/norain')
            elif opt.dataset_type == 'rain100':
                self.dir_Aderain = os.path.join(opt.dataroot, 'rain100H/train/rain')
                self.dir_Bderain = os.path.join(opt.dataroot, 'rain100H/train/norain') 

            elif opt.dataset_type == 'noise25':
                self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train_new_25/input.flist')
                self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train_new_25/gt.flist')
            elif opt.dataset_type == 'noise50':
                self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train_new_50/input.flist')
                self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train_new_50/gt.flist')                
            elif opt.dataset_type == 'noise75':
                self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train_new_75/input.flist')
                self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train_new_75/gt.flist')
            elif opt.dataset_type == 'noise100':
                self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train_new/input.flist')
                self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train_new/gt.flist')

            elif opt.dataset_type == 'noise_multi':
                self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/multi_domain_noise/input_multi.flist')
                self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/multi_domain_noise/gt_multi.flist')
                
            elif opt.dataset_type == 'new3_1':
                # Rain_all
                self.dir_Aderain = os.path.join(opt.dataroot, 'Rain100L/rain_20')
                self.dir_Bderain = os.path.join(opt.dataroot, 'Rain100L/norain_20')
                # denoise
                self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train_new_5/input.flist')
                self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train_new_5/gt.flist')
                # dehazing
                self.dir_Ahaze = '/root/data1/linziyue/RDDM/data/OTS/haze_1.flist'
                self.dir_Bhaze = '/root/data1/linziyue/RDDM/data/OTS/clear_1.flist'
            elif opt.dataset_type == 'new3_25':
                # Rain_all
                self.dir_Aderain = os.path.join(opt.dataroot, 'Rain100L/rain_25')
                self.dir_Bderain = os.path.join(opt.dataroot, 'Rain100L/norain_25')
                # denoise
                self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train_new_25/input.flist')
                self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train_new_25/gt.flist')
                # dehazing
                self.dir_Ahaze = '/root/data1/linziyue/RDDM/data/OTS/haze_25.flist'
                self.dir_Bhaze = '/root/data1/linziyue/RDDM/data/OTS/clear_25.flist'
            elif opt.dataset_type == 'new3_50':
                # Rain_all
                self.dir_Aderain = os.path.join(opt.dataroot, 'Rain100L/rain_50')
                self.dir_Bderain = os.path.join(opt.dataroot, 'Rain100L/norain_50')
                # denoise
                self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train_new_50/input.flist')
                self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train_new_50/gt.flist')
                # dehazing
                self.dir_Ahaze = '/root/data1/linziyue/RDDM/data/OTS/haze_50.flist'
                self.dir_Bhaze = '/root/data1/linziyue/RDDM/data/OTS/clear_50.flist'
            elif opt.dataset_type == 'new3_75':
                # Rain_all
                self.dir_Aderain = os.path.join(opt.dataroot, 'Rain100L/rain_75')
                self.dir_Bderain = os.path.join(opt.dataroot, 'Rain100L/norain_75')
                # denoise
                self.dir_Anoise = os.path.join(opt.dataroot, 'denoise/train_new_75/input.flist')
                self.dir_Bnoise = os.path.join(opt.dataroot, 'denoise/train_new_75/gt.flist')
                # dehazing
                self.dir_Ahaze = '/root/data1/linziyue/RDDM/data/OTS/haze_75.flist'
                self.dir_Bhaze = '/root/data1/linziyue/RDDM/data/OTS/clear_75.flist'       
        else:
            self.dir_Alol = os.path.join(opt.dataroot, 'LOL/eval15/low')
            self.dir_Blol = os.path.join(opt.dataroot, 'LOL/eval15/high')
            
            # self.dir_Aderain = os.path.join(opt.dataroot, 'syn_rain/test/Rain_all/data')
            # self.dir_Bderain = os.path.join(opt.dataroot, 'syn_rain/test/Rain_all/gt')
            # self.dir_Aderain = os.path.join(opt.dataroot, 'syn_rain/test/Test1200/input')
            # self.dir_Bderain = os.path.join(opt.dataroot, 'syn_rain/test/Test1200/target')
            self.dir_Aderain = os.path.join(opt.dataroot, 'syn_rain/test/Rain100H/input')
            self.dir_Bderain = os.path.join(opt.dataroot, 'syn_rain/test/Rain100H/target')

            # self.dir_Ainpaint = '/root/data1/linziyue/RDDM/data/CelebA_HQ_mask/irregular'
            # self.dir_Binpaint = '/root/data1/linziyue/RDDM/data/CelebA_HQ_256/train'

            self.dir_Ainpaint = '/root/data1/houjiahe/Decouple_RDDM/DIV2K/DIV2K_train_LR_bicubic/X4_sub_480'
            self.dir_Binpaint = '/root/data1/houjiahe/Decouple_RDDM/DIV2K/DIV2K_train_HR_sub'
            if opt.dataset_type == 'rain100L':
                self.dir_Aderain = os.path.join(opt.dataroot, 'syn_rain/test/Rain100L/input')
                self.dir_Bderain = os.path.join(opt.dataroot, 'syn_rain/test/Rain100L/target')
            # self.dir_Aderain = os.path.join(opt.dataroot, 'rain100H/test/rain')
            # self.dir_Bderain = os.path.join(opt.dataroot, 'rain100H/test/norain')
            # self.dir_Aderain = os.path.join(opt.dataroot, 'syn_rain/test/Test1200/input')
            # self.dir_Bderain = os.path.join(opt.dataroot, 'syn_rain/test/Test1200/target')

            self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/test/GoPro/input')
            self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/test/GoPro/target')

            # self.dir_Ablur = os.path.join(opt.dataroot, 'Deblur/test/GoPro/subset/input')
            # self.dir_Bblur = os.path.join(opt.dataroot, 'Deblur/test/GoPro/subset/target')
            # self.dir_Asnow = os.path.join(opt.dataroot, 'desnow/CSD2/Train/Snow')
            # self.dir_Bsnow = os.path.join(opt.dataroot, 'desnow/CSD2/Train/Gt')
                                                            
            self.dir_Asnow = os.path.join(opt.dataroot, 'snow100k/smol_Test/synthetic')
            self.dir_Bsnow = os.path.join(opt.dataroot, 'snow100k/smol_Test/gt')
            self.dir_Ahaze = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/hazy.flist')
            self.dir_Bhaze = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/gt.flist')
            # self.dir_Ahaze = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/hazy_20.flist')
            # self.dir_Bhaze = os.path.join(opt.dataroot, 'RESIDE/SOTS/outdoor/gt_20.flist')
            self.dir_Anoise1 = os.path.join(opt.dataroot, 'denoise/test/CBSD68_test25')
            self.dir_Bnoise1 = os.path.join(opt.dataroot, 'denoise/test/CBSD68')
            self.dir_Anoise2 = os.path.join(opt.dataroot, 'denoise/test/Kodak_test50')
            self.dir_Bnoise2 = os.path.join(opt.dataroot, 'denoise/test/Kodak')
            self.dir_Anoise3 = os.path.join(opt.dataroot, 'denoise/test/Urban100_test25')
            self.dir_Bnoise3 = os.path.join(opt.dataroot, 'denoise/test/Urban100')
            self.dir_Anoise4 = os.path.join(opt.dataroot, 'denoise/test/McMaster_test50')
            self.dir_Bnoise4 = os.path.join(opt.dataroot, 'denoise/test/McMaster')
            self.dir_Anoise5 = os.path.join(opt.dataroot, 'denoise/test/mult-domain_test_mid_noise.flist')
            self.dir_Bnoise5 = os.path.join(opt.dataroot, 'denoise/test/mult-domain_test_mid_gt.flist')
            self.dir_Anoise6 = os.path.join(opt.dataroot, 'denoise/test/CBSD68_test_mid_noise.flist')
            self.dir_Bnoise6 = os.path.join(opt.dataroot, 'denoise/test/CBSD68_test_mid_gt.flist')
            
            if opt.dataset_type == 'old':
                # self.dir_Asnow = os.path.join(opt.dataroot, 'snow100k/Snow100K-L/synthetic')
                # self.dir_Bsnow = os.path.join(opt.dataroot, 'snow100k/Snow100K-L/gt')
                self.dir_Asnow = os.path.join(opt.dataroot, 'snow100k/small_Test/synthetic')
                self.dir_Bsnow = os.path.join(opt.dataroot, 'snow100k/small_Test/gt')

        # 任务分支配置
        if task == 'light':
            self.A_paths = sorted(make_dataset(self.dir_Alol, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Blol, opt.max_dataset_size))
        elif task == 'rain':
            self.A_paths = sorted(make_dataset(self.dir_Aderain, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bderain, opt.max_dataset_size))
        elif task == 'snow':
            self.A_paths = sorted(make_dataset(self.dir_Asnow, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bsnow, opt.max_dataset_size))
        elif task == 'blur':
            self.A_paths = sorted(make_dataset(self.dir_Ablur, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bblur, opt.max_dataset_size))
        elif task == 'fog':
            self.A_paths = make_dataset_from_flist(self.dir_Ahaze, opt.max_dataset_size)
            self.B_paths = make_dataset_from_flist(self.dir_Bhaze, opt.max_dataset_size)
        # 修改：inpaint任务使用随机配对
        elif task == 'inpaint':
            print('task == inpaint')
            self.A_paths = sorted(make_dataset(self.dir_Ainpaint, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Binpaint, opt.max_dataset_size))
            # 标记为随机配对模式
        elif task == 'trans':
            self.A_paths = sorted(make_dataset(self.dir_Atrans, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Btrans, opt.max_dataset_size))
        elif task == 'noise_flist':  # 专用于 noise 13k 减量数据集训练使用
            self.A_paths = make_dataset_from_flist(self.dir_Anoise, opt.max_dataset_size)
            self.B_paths = make_dataset_from_flist(self.dir_Bnoise, opt.max_dataset_size)
        elif task == 'noise':
            self.A_paths = sorted(make_dataset(self.dir_Anoise, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bnoise, opt.max_dataset_size))
            # self.A_paths = make_dataset_from_flist(self.dir_Anoise, opt.max_dataset_size)
            # self.B_paths = make_dataset_from_flist(self.dir_Bnoise, opt.max_dataset_size)
        elif task == 'noise1':
            self.A_paths = sorted(make_dataset(self.dir_Anoise1, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bnoise1, opt.max_dataset_size))
        elif task == 'noise2':
            self.A_paths = sorted(make_dataset(self.dir_Anoise2, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bnoise2, opt.max_dataset_size))
        elif task == 'noise3':
            self.A_paths = sorted(make_dataset(self.dir_Anoise3, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bnoise3, opt.max_dataset_size))
        elif task == 'noise4':
            self.A_paths = sorted(make_dataset(self.dir_Anoise4, opt.max_dataset_size))
            self.B_paths = sorted(make_dataset(self.dir_Bnoise4, opt.max_dataset_size))
        elif task == 'noise5':
            self.A_paths = make_dataset_from_flist(self.dir_Anoise5, opt.max_dataset_size)
            self.B_paths = make_dataset_from_flist(self.dir_Bnoise5, opt.max_dataset_size)
        elif task == 'noise6':
            self.A_paths = make_dataset_from_flist(self.dir_Anoise6, opt.max_dataset_size)
            self.B_paths = make_dataset_from_flist(self.dir_Bnoise6, opt.max_dataset_size)
        else:
            raise ValueError(f"No dataset found for the task: '{task}'. "
                             f"Please check your task name and dataset path.")

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        
        # 修改：根据配对模式打印不同的信息
        if hasattr(self, 'random_pairing') and self.random_pairing:
            print(f"Obtain {task} input dataset with size: {self.A_size} (random pairing)")
            print(f"Obtain {task} target dataset with size: {self.B_size} (random pairing)")
        else:
            print(f"Obtain {task} input dataset with size: {self.A_size}")
            print(f"Obtain {task} target dataset with size: {self.B_size}")
            
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # 修改：根据配对模式选择不同的路径获取方式

        # 成对配对：使用相同索引
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        condition = Image.open(A_path).convert('RGB')  # condition
        gt = Image.open(B_path).convert('RGB')  # gt

        if 'LOL' in A_path or 'LSRW' in A_path:
            condition = cv2.cvtColor(np.asarray(condition), cv2.COLOR_RGB2BGR)
            gt = cv2.cvtColor(np.asarray(gt), cv2.COLOR_RGB2BGR)

            if self.crop_patch:
                gt, condition = self.get_patch([gt, condition], self.image_size)
            if 'LOL' in A_path:
                condition = self.cv2equalizeHist(condition) if self.equalizeHist else condition
            else:
                condition = condition

            images = [[gt, condition]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            g = p.generator(batch_size=1)
            augmented_images = next(g)
            gt = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
            condition = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB)

            gt = self.to_tensor(gt)
            condition = self.to_tensor(condition)
        else:
            w, h = condition.size
            transform_params = get_params(self.opt, condition.size)
            A_transform = get_transform(self.opt, transform_params, grayscale=False)
            B_transform = get_transform(self.opt, transform_params, grayscale=False)
            condition = A_transform(condition)
            gt = B_transform(gt)

            if self.opt.phase == 'train':
                if h < self.opt.crop_size or w < self.opt.crop_size:
                    osize = [self.opt.crop_size, self.opt.crop_size]
                    resi = transforms.Resize(osize, transforms.InterpolationMode.BICUBIC)
                    condition = resi(condition)
                    gt = resi(gt)

        return {'adap': condition, 'gt': gt, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            # condition
            name = self.input[index]
            if sub_dir == 0:
                return os.path.basename(name)
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir + "_" + os.path.basename(name)

    def get_patch(self, image_list, patch_size):
        i = 0
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h - patch_size)
        cc = random.randint(0, w - patch_size)
        for img in image_list:
            image_list[i] = img[rr:rr + patch_size, cc:cc + patch_size, :]
            i += 1
        return image_list

    def pad_img(self, img_list, patch_size, block_size=8):
        i = 0
        for img in img_list:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            bottom = 0
            right = 0
            if h < patch_size:
                bottom = patch_size - h
                h = patch_size
            if w < patch_size:
                right = patch_size - w
                w = patch_size
            bottom = bottom + (h // block_size) * block_size + \
                     (block_size if h % block_size != 0 else 0) - h
            right = right + (w // block_size) * block_size + \
                    (block_size if w % block_size != 0 else 0) - w
            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            i += 1
        return img_list

    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.input[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size - h
            h = patch_size
        if w < patch_size:
            right = patch_size - w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + \
                 (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + \
                (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]