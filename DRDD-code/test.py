import os
import sys
import torch
import yaml
import argparse

from data.universal_dataset import AlignedDataset_all
from src.drdd import (
    ResidualDiffusion,
    Trainer, Unet,
    UnetRes, set_seed
)
from src.guided_diffusion import dist_util, logger
from src.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

CONFIG_PATH = "./config/"
sys.stdout.flush()
set_seed(10)

import torch
from thop import profile, clever_format

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default='test')
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    parser.add_argument('--load_size', type=int, default=268, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--preprocess', type=str, default='none',
                        help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_true', default=True,
                   help='disable flipping (enabled by default for test phase)')
    parser.add_argument('--data_config', type=str, required=True, help="Name of the data config")
    parser.add_argument('--dataset_type', type=str, default="old", help='use new | old dataset')

    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    add_dict_to_argparser(parser, defaults)

    opt = parser.parse_args()

    with open(CONFIG_PATH + opt.data_config + ".yaml", "r") as f:
        config = yaml.safe_load(f)

    opt.image_size=config['noise_model']['image_size']
    opt.num_channels=config['noise_model']['num_channels']
    opt.channel_mult=config['noise_model']['channel_mult']
    opt.num_res_blocks=config['noise_model']['num_res_blocks']
    opt.dataroot = config['data']['data_root']

    print(opt)
    return opt, config


opt, config = parser_args()

dataset = AlignedDataset_all(opt, augment_flip=False, equalizeHist=True, crop_patch=False, generation=False,
                             task=config['test']['task'])

res_model = UnetRes(**config['res_model'])

logger.log("creating model and diffusion...")

noise_model, noise_diffusion = create_model_and_diffusion(
    **args_to_dict(opt, model_and_diffusion_defaults().keys())
)

diffusion = ResidualDiffusion(
    res_model,
    noise_model,
    image_size=opt.image_size,
    timesteps=1000,
    delta_end=config['diffusion']['delta_end'],
    res_sampling_timesteps=config['sampling']['res_sampling_timesteps'],
    noise_sampling_timesteps=config['sampling']['noise_sampling_timesteps'],
    condition=config['diffusion']['condition'],
    sum_scale=config['diffusion']['sum_scale'],
    test_res_or_noise=config['diffusion']['test_res_or_noise']
)

trainer = Trainer(
    diffusion,
    dataset,
    opt,
    num_samples=config['test']['num_samples'],
    amp=False,  # turn on mixed precision
    convert_image_to="RGB",
    results_folder=config['test']['results_ckpt'],
    condition= True,
    num_unet= 2,
)
trainer.load()

trainer.set_results_folder(os.path.join(config['test']['test_result_folder'], str("test")))
with torch.no_grad():
    result = trainer.test(last=config['test']['last'])