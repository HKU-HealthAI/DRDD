import os
import sys
import torch
import yaml
import argparse
import torch
from thop import profile, clever_format
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from data.universal_dataset import AlignedDataset_all
from src.unet_plus_decouple_noIn import (
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

CONFIG_PATH = "./configs/"
sys.stdout.flush()
set_seed(10)

def calculate_flops(model, input_size=(1, 6, 256, 256), time_dim=1000):
    """计算需要time参数的模型的FLOPs"""
    model.eval()
    
    # 创建随机输入和时间步长
    input_tensor = torch.randn(input_size)
    time_tensor = torch.randint(0, time_dim, (input_size[0],))
    
    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=(input_tensor, time_tensor), verbose=False)
    
    # 格式化输出
    flops, params = clever_format([flops, params], "%.3f")
    
    return flops, params
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default='train')
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    parser.add_argument('--load_size', type=int, default=134, help='scale images to this size')  # 134,268
    parser.add_argument('--crop_size', type=int, default=128, help='then crop to this size')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--preprocess', type=str, default='crop',
                        help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_true',
                        help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--data_config', type=str, default="train_diffuir")
    parser.add_argument('--dataset', type=str, default="3dataset", help='all | nofog | nosnow | 3dataset| rain100')
    parser.add_argument('--dataset_type', type=str, default="old", help='use new | old dataset')

    defaults = dict(
        weight_decay=0.0,
        lr_anneal_steps=0,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    add_dict_to_argparser(parser, defaults)

    opt = parser.parse_args()

    print(opt)

    with open(CONFIG_PATH + opt.data_config + ".yaml", "r") as f:
        config = yaml.safe_load(f)

    opt.num_channels=config['noise_model']['num_channels']
    opt.channel_mult=config['noise_model']['channel_mult']
    opt.num_res_blocks=config['noise_model']['num_res_blocks']
    

    opt.bsize = config['training']['bsize']
    opt.image_size = config['noise_model']['image_size']
    opt.dataroot = config['data']['data_root']
    opt.noise_lr = config['noise_model']['lr']
    opt.res_lr = config['res_model']['lr']
    
    return opt, config


opt, config = parser_args()

dataset_fog = AlignedDataset_all(opt, augment_flip=True, equalizeHist=True, crop_patch=True,
                                 generation=False, task='fog')
dataset_light = AlignedDataset_all(opt, augment_flip=True, equalizeHist=True, crop_patch=True,
                                   generation=False, task='light')
dataset_rain = AlignedDataset_all(opt, augment_flip=True, equalizeHist=True, crop_patch=True,
                                  generation=False, task='rain')
dataset_snow = AlignedDataset_all(opt, augment_flip=True, equalizeHist=True, crop_patch=True,
                                  generation=False, task='snow')
dataset_blur = AlignedDataset_all(opt, augment_flip=True, equalizeHist=True, crop_patch=True,
                                  generation=False, task='blur')
dataset_noise = AlignedDataset_all(opt, augment_flip=True, equalizeHist=True, crop_patch=True,
                                  generation=False, task='noise_flist')
dataset_inpaint = AlignedDataset_all(opt, augment_flip=True, equalizeHist=True, crop_patch=True,
                                  generation=False, task='inpaint')
dataset_trans = AlignedDataset_all(opt, augment_flip=True, equalizeHist=True, crop_patch=True,
                                  generation=False, task='trans')

dataset = [dataset_fog, dataset_light, dataset_rain, dataset_snow, dataset_blur, dataset_noise, dataset_inpaint, dataset_trans]
# dataset = [dataset_fog, dataset_light, dataset_rain, dataset_snow, dataset_blur, dataset_noise]
logger.log("creating model and diffusion...")
noise_model, noise_diffusion = create_model_and_diffusion(
    **args_to_dict(opt, model_and_diffusion_defaults().keys())
)
if config['noise_model']['pretrain_model'] != None:

    checkpoint = torch.load(config['noise_model']['pretrain_model'], map_location="cpu")
    checkpoint["out.2.weight"] = checkpoint["out.2.weight"][:3]  # 取前3个输出通道
    checkpoint["out.2.bias"] = checkpoint["out.2.bias"][:3]
    checkpoint.pop("input_blocks.0.0.weight", None)
    checkpoint.pop("input_blocks.0.0.bias", None)
    # 使用方式
    noise_model.load_state_dict(checkpoint, strict=False)  # 现在可以加载了
    missing_keys = set(checkpoint.keys()) - set(noise_model.state_dict().keys())
    print(f"未加载的层：{missing_keys}")


# -------------------------------------------------------------------------------#
diff_config = config['diffusion']
diffusion = ResidualDiffusion(
    UnetRes(**config['res_model']),
    noise_model,
    image_size=opt.image_size,
    timesteps=1000,
    delta_end=diff_config['delta_end'],
    norm=diff_config['norm'],
    norm_lambda=diff_config['norm_lambda'],
    sampling_timesteps=config['training']['sampling_timesteps'],
    condition=diff_config['condition'],
    sum_scale=diff_config['sum_scale'],
    test_res_or_noise=diff_config['test_res_or_noise']
)

total_params = sum(p.numel() for p in diffusion.parameters())
print(f"模型总参数量: {total_params}")

res_model = UnetRes(**config['res_model'])
print("计算Res模型的FLOPs...")
res_flops, res_params = calculate_flops(res_model)
print(f"Res模型 - FLOPs: {res_flops}, 参数量: {res_params}")

print("计算Noise模型的FLOPs...")
noise_flops, noise_params = calculate_flops(noise_model)
print(f"Noise模型 - FLOPs: {noise_flops}, 参数量: {noise_params}")

train_config = config['training']
trainer = Trainer(
    diffusion,
    dataset,
    opt,
    num_samples=train_config['num_samples'],
    noise_lr=opt.noise_lr,
    res_lr=opt.res_lr,
    train_num_steps=train_config['train_num_steps'],  # total training steps
    gradient_accumulate_every=train_config['gradient_accumulate'],  # gradient accumulation steps
    ema_decay=train_config['ema_decay'],  # exponential moving average decay
    amp=False,  # turn on mixed precision
    convert_image_to="RGB",
    results_folder=train_config['results_folder'],
    condition=train_config['condition'],
    save_and_sample_every=train_config['save_and_sample_every'],
    num_unet=train_config['num_unet'],
    fp16=opt.use_fp16
)

# train
# 多卡模式下注意，防止进程冲突
#if not trainer.accelerator.is_main_process:
#    pass
#else:
#    trainer.set_results_folder(train_config['results_folder'])
#    trainer.load(80)
#trainer.accelerator.wait_for_everyone()
if config['training']['load'] != None:
    trainer.load(config['training']['load'])
trainer.train()

