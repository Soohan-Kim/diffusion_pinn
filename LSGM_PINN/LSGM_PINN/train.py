
import sys
from functools import partial

sys.path.append("/home/work/.local/bin/")
sys.path.append("/home/work/.local/lib/python3.7/site-packages/ninja-1.11.1.dist-info/")
sys.path.append("/home/work/.local/lib/python3.7/site-packages/ninja/")

import torch
import torchvision
import torchvision.transforms as transforms

from functorch import jacfwd, vmap
#from torch.func import jacfwd, vmap, jacrev

import random

from torch.optim import RAdam
from torch.optim.lr_scheduler import LambdaLR
import time

import torch.multiprocessing as mp
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import MultivariateNormal
from torch.autograd.functional import jacobian
from torch.autograd import grad

import tqdm
#import pickle
import numpy as np

from PIL import Image

import re
from collections import OrderedDict

import model_configs
import json

sys.path.append('./LSGM')

from LSGM.nvae import NVAE
from LSGM.diffusion_continuous import make_diffusion
from LSGM.util import utils
from torch.optim import Adam as FusedAdam
from LSGM.util.ema import EMA

# (1) Load pretrained VAE, DAE from https://github.com/NVlabs/LSGM
# referencing evaluate_vada.py

# (2) Sample Z_t from VP-SDE forward process conditional distribution in latent space
# forward through pretrained VAE encoder with true image and then perform forward process sampling
# referencing training_obj_disjoint.py -> train_vada_disjoint

# (3) Form boundary condition loss with true image and generated sample starting from Z_t
# use our network and pretrained VAE decoder to generate sample starting from Z_t
# referencing evaluate_diffusion.py -> generate_samples_vada

# (4) Form PINN loss with Z_t and pretrained DAE
# use patch-wise calculation if needed for large-size latent spaces 

# (5) Backpropagation

if __name__ == "__main__":
    pretrained_checkpoint = './pretrained/checkpoint.pt'

    checkpoint = torch.load(pretrained_checkpoint)
    args = checkpoint['args']

    # adding some arguments for backward compatibility.
    if not hasattr(args, 'num_x_bits'):
        #logging.info('*** Setting %s manually ****', 'num_x_bits')
        setattr(args, 'num_x_bits', 8)

    if not hasattr(args, 'channel_mult'):
        #logging.info('*** Setting %s manually ****', 'channel_mult')
        setattr(args, 'channel_mult', [1, 2])

    if not hasattr(args, 'mixing_logit_init'):
        #logging.info('*** Setting %s manually ****', 'mixing_logit_init')
        setattr(args, 'mixing_logit_init', -3.0)

    arch_instance_nvae = utils.get_arch_cells(args.arch_instance, args.use_se)

    vae = NVAE(args, arch_instance_nvae)
    vae.load_state_dict(checkpoint['vae_state_dict'])

    num_input_channels = vae.latent_structure()[0]
    dae = utils.get_dae_model(args, num_input_channels)
    dae.load_state_dict(checkpoint['dae_state_dict'])
    
    # checkpoint.pt models require swapping EMA parameters
    dae_optimizer = FusedAdam(dae.parameters(), args.learning_rate_dae,
                                weight_decay=args.weight_decay, eps=1e-4)
    # add EMA functionality to the optimizer
    dae_optimizer = EMA(dae_optimizer, ema_decay=args.ema_decay)
    dae_optimizer.load_state_dict(checkpoint['dae_optimizer'])

    # replace DAE parameters with EMA values
    dae_optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    vae.eval()
    dae.eval()

    print(dae.num_input_channels, dae.input_size, dae.input_size)