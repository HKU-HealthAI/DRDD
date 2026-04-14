import copy
import glob
import math
import os
import random
import time as time_module
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
import torchvision.transforms as transforms
import Augmentor
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from PIL import Image
import time
from torch import einsum, nn
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import utils
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from thop import profile
import copy
import importlib
import lpips

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])
# helpers functions
metric_module = importlib.import_module('metrics')


def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions


def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5


# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)



class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered



class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            condition=False,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.depth = len(dim_mults)
        input_channels = channels + channels * (1 if condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x, time):
        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for i, (block1, block2, attn, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W].contiguous()
        return x


class UnetRes(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            resnet_block_groups=8,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
            condition=False,
            test_res_or_noise="res_noise",
            **kwargs
    ):
        super().__init__()
        self.condition = condition
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.test_res_or_noise = test_res_or_noise
        # determine dimensions

        self.unet = Unet(dim,
                         init_dim=init_dim,
                         out_dim=out_dim,
                         dim_mults=dim_mults,
                         channels=channels,
                         resnet_block_groups=resnet_block_groups,
                         learned_variance=learned_variance,
                         learned_sinusoidal_cond=learned_sinusoidal_cond,
                         random_fourier_features=random_fourier_features,
                         learned_sinusoidal_dim=learned_sinusoidal_dim,
                         condition=condition)

    def forward(self, x, time):
        return self.unet(x, time)


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))



class ResidualDiffusion(nn.Module):
    def __init__(
            self,
            res_model,
            noise_model,
            *,
            image_size,
            timesteps=1000,
            delta_end=1.5e-3,
            norm="None",
            norm_lambda=(1e-7, 1e-5), 
            res_sampling_timesteps=None,
            noise_sampling_timesteps=None,
            ddim_sampling_eta=0.,
            condition=False,
            sum_scale=None,
            test_res_or_noise="None",
    ):
        super().__init__()
        assert not (
                type(self) == ResidualDiffusion and res_model.channels != res_model.out_dim)
        assert not res_model.random_or_learned_sinusoidal_cond

        self.res_model = res_model
        self.noise_model = noise_model
        self.channels = self.res_model.channels
        self.image_size = image_size
        self.condition = condition
        self.test_res_or_noise = test_res_or_noise
        self.norm = norm
        self.norm_lambda = norm_lambda

        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumsum = 1 - alphas_cumprod ** 0.5
        betas2_cumsum = 1 - alphas_cumprod

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        alphas = alphas_cumsum - alphas_cumsum_prev
        alphas[0] = 0
        betas2 = betas2_cumsum - betas2_cumsum_prev
        betas2[0] = 0

        betas_cumsum = torch.sqrt(betas2_cumsum)

        posterior_variance = betas2 * betas2_cumsum_prev / betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.res_sampling_timesteps = default(res_sampling_timesteps, timesteps)
        self.noise_sampling_timesteps = default(noise_sampling_timesteps, timesteps)


        assert self.res_sampling_timesteps <= timesteps or self.noise_sampling_timesteps <= timesteps
        self.is_ddim_sampling = True
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val):
            return self.register_buffer(
                name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1 - alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev / betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                                                 alphas_cumsum_prev - betas2_cumsum_prev * alphas) / betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2 / betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def init(self):
        timesteps = 1000

        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(
            beta_start, beta_end, timesteps, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumsum = 1 - alphas_cumprod ** 0.5
        betas2_cumsum = 1 - alphas_cumprod

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        alphas = alphas_cumsum - alphas_cumsum_prev
        alphas[0] = alphas[1]
        betas2 = betas2_cumsum - betas2_cumsum_prev
        betas2[0] = betas2[1]

        betas_cumsum = torch.sqrt(betas2_cumsum)

        posterior_variance = betas2 * betas2_cumsum_prev / betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        self.alphas = alphas
        self.alphas_cumsum = alphas_cumsum
        self.one_minus_alphas_cumsum = 1 - alphas_cumsum
        self.betas2 = betas2
        self.betas = torch.sqrt(betas2)
        self.betas2_cumsum = betas2_cumsum
        self.betas_cumsum = betas_cumsum
        self.posterior_mean_coef1 = betas2_cumsum_prev / betas2_cumsum
        self.posterior_mean_coef2 = (
                                            betas2 * alphas_cumsum_prev - betas2_cumsum_prev * alphas) / betas2_cumsum
        self.posterior_mean_coef3 = betas2 / betas2_cumsum
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_x_noisy_from_res_noise(self, x_t, t, x_res, x_input):
        return (
                x_t - extract(self.alphas_cumsum, t, x_t.shape) * x_res

        )

    def predict_start_from_noise(self, x_t, t, noise, x_input):
        return (
                x_t - extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
                extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
                extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_noise_predictions(self, x_input, x, t, x_input_condition=0, x_self_cond=None, clip_denoised=False):
        if not self.condition:
            x_in = x
        else:
            x_in = torch.cat((x, x_input), dim=1)

        model_output = self.noise_model(x_in,
                                        self.betas_cumsum[t] * self.num_timesteps)
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_denoised else identity

        pred_res = 0
        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise, x_input)
        x_start = maybe_clip(x_start)
        return ModelResPrediction(pred_res, pred_noise, x_start)

    def model_res_predictions(self, x_input, x, t, x_input_condition=0, x_self_cond=None, clip_denoised=False):
        if not self.condition:
            x_in = x
        else:
            x_in = torch.cat((x, x_input), dim=1)
        model_output = self.res_model(x_in,
                                      self.alphas_cumsum[t] * self.num_timesteps)
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_denoised else identity

        pred_res = model_output
        pred_res = maybe_clip(pred_res)
        pred_noise = 0
        x_start = self.predict_x_noisy_from_res_noise(x, t, pred_res, x_input)
        x_start = maybe_clip(x_start)
        return ModelResPrediction(pred_res, pred_noise, x_start)

    @torch.no_grad()


    def ddim_sample(self, x_input, shape, last=True, task=None):
            x_input = x_input[0]

            batch  = shape[0]
            device = self.betas.device
            T      = self.num_timesteps
            eta    = self.ddim_sampling_eta

            N_RES_STEPS   = self.res_sampling_timesteps   
            N_NOISE_STEPS = self.noise_sampling_timesteps   

            def make_time_pairs(num_steps: int):

                times = torch.linspace(-1, T - 1, steps=num_steps + 1)
                times = list(reversed(times.to(torch.int64).tolist()))
                return list(zip(times[:-1], times[1:]))

            time_pairs_res   = make_time_pairs(N_RES_STEPS)
            time_pairs_noise = make_time_pairs(N_NOISE_STEPS)


            if self.condition:

                gauss_noise = math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
                img = x_input + gauss_noise
                input_add_noise = img
            else:
                img = torch.randn(shape, device=device)

            x_start = None
            use_pred = "use_pred_noise"

            if not last:
                img_list = []

            for time, time_next in time_pairs_res:
                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

                preds      = self.model_res_predictions(x_input, img, time_cond, task)
                pred_res   = preds.pred_res
                pred_noise = preds.pred_noise
                x_start    = preds.pred_x_start

                if time_next < 0:
                    img = x_start
                    if not last:
                        img_list.append(img)
                    continue

                alpha_cumsum       = self.alphas_cumsum[time]
                alpha_cumsum_next  = self.alphas_cumsum[time_next]
                alpha              = alpha_cumsum - alpha_cumsum_next

                betas2_cumsum      = self.betas2_cumsum[time]
                betas2_cumsum_next = self.betas2_cumsum[time_next]
                betas2             = betas2_cumsum - betas2_cumsum_next
                betas              = betas2.sqrt()
                betas_cumsum       = self.betas_cumsum[time]
                betas_cumsum_next  = self.betas_cumsum[time_next]

                sigma2 = eta * (betas2 * betas2_cumsum_next / betas2_cumsum)

                noise = 0 if eta == 0 else torch.randn_like(img)

                if use_pred == "use_pred_noise":
                    img = img - alpha * pred_res + sigma2.sqrt() * noise

                if not last:
                    img_list.append(img)

            for time, time_next in time_pairs_noise:            
                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

                preds      = self.model_noise_predictions(x_input, img, time_cond, task)
                pred_res   = preds.pred_res
                pred_noise = preds.pred_noise
                x_start    = preds.pred_x_start

                if time_next < 0:
                    img = x_start
                    if not last:
                        img_list.append(img)
                    continue

                alpha_cumsum       = self.alphas_cumsum[time]
                alpha_cumsum_next  = self.alphas_cumsum[time_next]
                alpha              = alpha_cumsum - alpha_cumsum_next

                betas2_cumsum      = self.betas2_cumsum[time]
                betas2_cumsum_next = self.betas2_cumsum[time_next]
                betas2             = betas2_cumsum - betas2_cumsum_next
                betas              = betas2.sqrt()
                betas_cumsum       = self.betas_cumsum[time]
                betas_cumsum_next  = self.betas_cumsum[time_next]

                sigma2 = eta * (betas2 * betas2_cumsum_next / betas2_cumsum)

                noise = 0 if eta == 0 else torch.randn_like(img)

                if use_pred == "use_pred_noise":
                    img = img - (betas_cumsum - (betas2_cumsum_next - sigma2).sqrt()) * pred_noise \
                        + sigma2.sqrt() * noise

                if not last:
                    img_list.append(img)
            
            if self.condition:
                if not last:
                    img_list = [input_add_noise] + img_list
                else:
                    img_list = [input_add_noise, img]
                return unnormalize_to_zero_to_one(img_list)
            else:
                if not last:
                    img_list = img_list
                else:
                    img_list = [img]
                return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def sample(self, x_input=0, batch_size=1, last=True, task=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        if self.condition:
            x_input = 2 * x_input - 1
            x_input = x_input.unsqueeze(0)

            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)
        return sample_fn(x_input, size, last=last, task=task)





class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset,
            opts,
            *,
            gradient_accumulate_every=1,
            augment_flip=True,
            noise_lr=1e-4,
            res_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            num_unet=1,
            num_samples=25,
            results_folder='./results/sample',
            amp=False,
            fp16=False,
            split_batches=True,
            convert_image_to=None,
            condition=False,
            sub_dir=False,
    ):
        
        super().__init__()
        print(f"noise_lr: {noise_lr}")
        print(f"res_lr: {res_lr}")
        print(f"results_folder: {results_folder}")
        print(f"opts: {opts}")
       
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.sub_dir = sub_dir
        self.accelerator.native_amp = amp
        self.num_unet = num_unet
        self.model = diffusion_model
        self.opts = opts

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition

        if self.condition:
            if opts.phase == "train":
                raise NotImplementedError("Training details are temporarily unavailable. Full training details will be released upon acceptance.")     
            else:
                self.sample_dataset = dataset

        # optimizer
        self.opt0 = Adam(diffusion_model.res_model.unet.parameters(), lr=res_lr, betas=adam_betas)
        self.opt1 = Adam(diffusion_model.noise_model.parameters(), lr=noise_lr, betas=adam_betas)
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)

            self.set_results_folder(results_folder)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt0, self.opt1 = self.accelerator.prepare(
            self.model, self.opt0, self.opt1)

        device = self.accelerator.device
        self.device = device
        
    def load(self):
        path = self.results_folder
        if path.exists():
            data = torch.load(str(path), map_location=self.device)
            self.model = self.accelerator.unwrap_model(self.model)

            self.model.load_state_dict(data['model'])
            self.step = data['step']

            self.opt0.load_state_dict(data['opt0'])
            self.opt0.param_groups[0]['capturable'] = True
            self.opt1.load_state_dict(data['opt1'])
            self.opt1.param_groups[0]['capturable'] = True
            self.ema.load_state_dict(data['ema'])

            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

            print("load model - " + str(path))


    def train(self):
        pass

    def test(self, sample=False, last=True):
        self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")
        if self.condition:
            self.ema.ema_model.eval()
            loader = DataLoader(
                dataset=self.sample_dataset,
                batch_size=1)
            i = 0
            cnt = 0
            opt_metric = {
                'psnr': {
                    'type': 'calculate_psnr',
                    'crop_border': 0,
                    'test_y_channel': True
                },
                'ssim': {
                    'type': 'calculate_ssim',
                    'crop_border': 0,
                    'test_y_channel': True
                }
            }
            self.metric_results = {
                metric: 0
                for metric in opt_metric.keys()
            }

            lpips_fn = lpips.LPIPS(net='alex').to(self.device)
            self.metric_results['lpips'] = 0
            tran = transforms.ToTensor()

            pbar = tqdm(loader, desc="Test Processing")
            for items in pbar:
                if self.condition:
                    file_ = items["A_paths"][0]
                    file_name = file_.split('/')[-4]
                else:
                    file_name = f'{i}.png'

                i += 1

                with torch.no_grad():
                    batches = self.num_samples

                    data = items
                    x_input_sample = data["adap"].to(self.device)
                    gt = data["gt"].to(self.device)

                    all_images_list = list(self.ema.ema_model.sample(
                        x_input_sample, batch_size=batches, last=last, task=file_))
                if last:
                    all_images_list = [all_images_list[-1]]
                    all_images = torch.cat(all_images_list, dim=0)
                else:
                    all_images_list.append(gt)
                    all_images = torch.cat(all_images_list, dim=0)
                
                if last:
                    nrow = int(math.sqrt(self.num_samples))
                else:
                    nrow = all_images.shape[0]
                save_path = str(self.results_folder / file_name)
                os.makedirs(save_path, exist_ok=True)
                full_path = os.path.join(save_path, file_.split('/')[-1]).replace('_fake_B', '')
                utils.save_image(all_images, full_path, nrow=nrow)
                pbar.set_postfix_str(f"test_save : {full_path}")

                # calculate the metric
                if not last:
                    all_images_list = [all_images_list[-2]]
                    all_images = torch.cat(all_images_list, dim=0)

                sr_img = tensor2img(all_images, rgb2bgr=True)
                gt_img = tensor2img(gt, rgb2bgr=True)
                opt_metric_ = {
                    'psnr': {
                        'type': 'calculate_psnr',
                        'crop_border': 0,
                        'test_y_channel': True
                    },
                    'ssim': {
                        'type': 'calculate_ssim',
                        'crop_border': 0,
                        'test_y_channel': True
                    }
                }
                for name, opt_ in opt_metric_.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)

                def img_to_lpips_tensor(img):
                    if img.dtype == 'uint8' or img.max() > 1:
                        img = img.astype('float32') / 255.0
                    tensor = tran(img).unsqueeze(0)  # (1,3,H,W)
                    tensor = tensor * 2 - 1
                    return tensor.to(self.device)

                sr_tensor = img_to_lpips_tensor(sr_img)
                gt_tensor = img_to_lpips_tensor(gt_img)
                with torch.no_grad():
                    lpips_score = lpips_fn(sr_tensor, gt_tensor).item()
                self.metric_results['lpips'] += lpips_score

                cnt += 1

            current_metric = {}
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric[metric] = self.metric_results[metric]

            return current_metric['psnr'], current_metric['ssim'], current_metric['lpips']

        print("test end")

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)
