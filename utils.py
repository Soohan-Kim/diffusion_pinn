import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

'''
Functionalities

(1) constructing sigmas from given choice of time discretization
(2) Return activation function based on config
(3) getting timestep embeddings (positional)
(4) calculating Empirical Characteristic Function Distance (ECFD) Loss
'''

def get_sigmas(config):
    '''Construct sigmas = time t's array from t_1=eps to t_N=T'''
    sigmas = []
    N = config['discretization_steps']
    eps = config['time_start_epsilon']
    T = config['time_end_T']
    if config['discretization'] == 'edm_style':
        rho = config['discretization_rho']
        for i in range(1, N+1):
            sigmas.append((eps**(1/rho) + (i-1)/(N-1)*(T**(1/rho) - eps**(1/rho)))**rho)
        
    return np.array(sigmas)

def get_act(config):
  """Get activation functions from the config file."""

  if config['activation_func'] == 'elu':
    return nn.ELU()
  elif config['activation_func'] == 'relu':
    return nn.ReLU()
  elif config['activation_func'] == 'lrelu':
    return nn.LeakyReLU(negative_slope=0.2)
  elif config['activation_func'] == 'swish':
    return nn.SiLU()
  else:
    raise NotImplementedError('activation function does not exist!')
  
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  # emb = math.log(2.) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
  # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb

def get_ECFD_Loss(X, Y, device):
  num_freqs = 8 # Number of frequency t's to sample
  sigma = 1.0 # Gaussian Distribution sigma to use for sampling t's

  t = torch.randn((num_freqs, X.size(-1)), dtype=torch.float).to(device) * sigma

  tX = torch.matmul(t, X.T)
  tY = torch.matmul(t, Y.T)

  cos_tX = (torch.cos(tX)).mean(1)
  sin_tX = (torch.sin(tX)).mean(1)
  cos_tY = (torch.cos(tY)).mean(1)
  sin_tY = (torch.sin(tY)).mean(1)

  loss = (cos_tX - cos_tY)**2 + (sin_tX - sin_tY)**2

  return loss.mean()