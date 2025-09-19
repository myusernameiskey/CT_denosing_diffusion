import torch
from torch._C import device
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import numpy as np
# import controllable_generation_TV

from utils import restore_checkpoint, clear
from pathlib import Path
from models import utils as mutils
from models import ncsnpp
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)

import sampling_input 

import datasets
import time
# for radon
# from physics.ct import CT
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import sampling
from torch.autograd import Variable
from sampling import *
###############################################
# Configurations
###############################################
problem = 'Dental'
config_name = 'Dental_256_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 2000
ckpt_num = 99
config_ckptpath_name = '2)smooth_21_1try'
N = num_scales

vol_name = 'PROJ_SKULL_PreMolarR_03_Slice'
root = Path(f'D:/우성_Denoising_Main_Vatech_Endo_fulldataset/De_noising_1st(high_dose)/De_noising_smooth_std_내컴/De_noising_smooth_Factor_21_내컴/Data/Use_data/Test/1)Same/in/{vol_name}')

# Parameters for the inverse problem
# Nview = 8
# det_spacing = 1.0
# size = 256
# det_count = int((size * (2 * torch.ones(1)).sqrt()).ceil())
# lamb = 0.04
# rho = 10
# freq = 1

if sde.lower() == 'vesde':
    from configs.ve import Dental_256_ncsnpp_continuous as configs
    ckpt_filename = f"assets/{config_ckptpath_name}/checkpoints/checkpoint_{ckpt_num}.pth"
    config = configs.get_config()
    config.model.num_scales = N
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sde.N = N
    sampling_eps = 1e-5
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
probability_flow = False
snr = 0.16
n_steps = 1

batch_size = 1
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
# config.data.image_size = 800
random_seed = 0

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)  ## model

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())

# Specify save directory for saving generated samples
save_root = Path(f'./results/{config_name}/{problem}')
save_root.mkdir(parents=True, exist_ok=True)





irl_types = ['input', 'recon', 'target']
for t in irl_types:
    if t == 'recon':
        save_root_f = save_root / t / 'progress'
        save_root_f.mkdir(exist_ok=True, parents=True)
    else:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)


print("Loading all data")


test_ds= datasets.create_dataloader(config,evaluation=True)
num_data_test = len(test_ds.dataset)

sampling_shape = (config.eval.batch_size,
                    config.data.num_channels,
                    config.data.image_size, config.data.image_size)


pc_sampler_input = sampling_input.get_pc_sampler_withI(sde=sde,
                                                        shape=sampling_shape,
                                                        predictor=predictor,
                                                        corrector=corrector,
                                                        inverse_scaler=inverse_scaler,
                                                        snr=config.sampling.snr,
                                                        n_steps=config.sampling.n_steps_each,
                                                        probability_flow=config.sampling.probability_flow,
                                                        continuous=config.training.continuous,
                                                        denoise=config.sampling.noise_removal,
                                                        eps=sampling_eps,
                                                        device=config.device)

# all_img = []
for titer, batch in enumerate(test_ds,0):
    just_name = titer
    in_img, target_img =  Variable(batch[0]), Variable(batch[1])
    in_img = in_img.to(config.device)
    target_img = target_img.to(config.device)  
    sample, n = pc_sampler_input(score_model, scaler(in_img), save_root)

    plt.imsave(os.path.join(save_root, 'input', f'{just_name}.png'), clear(in_img), cmap='gray')
    plt.imsave(os.path.join(save_root, 'target', f'{just_name}.png'), clear(target_img), cmap='gray')
    plt.imsave(os.path.join(save_root, 'recon', f'{just_name}.png'), clear(sample), cmap='gray')
    # all_img.append(in_img)

# all_img = torch.cat(all_img, dim=0)
# print(f"Data loaded shape : {all_img.shape}")











# pc_sampler = sampling.get_pc_sampler(sde=sde,
#                                 shape=sampling_shape,
#                                 predictor=predictor,
#                                 corrector=corrector,
#                                 inverse_scaler=inverse_scaler,
#                                 snr=config.sampling.snr,
#                                 n_steps=config.sampling.n_steps_each,
#                                 probability_flow=config.sampling.probability_flow,
#                                 continuous=config.training.continuous,
#                                 denoise=config.sampling.noise_removal,
#                                 eps=sampling_eps,
#                                 device=config.device)


# x = pc_sampler(score_model)



# x = sde.prior_sampling(img.shape).to(img.device)
# x1 = x[300,0,:,:]
# import torchvision.transforms as transforms
# tf =transforms.ToPILImage()
# xxx = tf(x1)
# xxx.show()



# x = sde.prior_sampling(all_img.shape).to(all_img.device)
# x1 = x[0,0,:,:]
# import torchvision.transforms as transforms
# tf =transforms.ToPILImage()
# xxx = tf(x1)
# xxx.show()



# # x2 = all_img[0,0,:,:]
# import torchvision.transforms as transforms
# tf =transforms.ToPILImage()
# xxx = tf(x.squeeze())
# xxx.show()



# in_img2 = tf(in_img.squeeze())
# in_img2.show()

