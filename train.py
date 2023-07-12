
import sys
from functools import partial
#from diffusers import UNet2DModel

sys.path.append("/home/work/.local/bin/")
sys.path.append("/home/work/.local/lib/python3.7/site-packages/ninja-1.11.1.dist-info/")
sys.path.append("/home/work/.local/lib/python3.7/site-packages/ninja/")

import torch
import torchvision
import torchvision.transforms as transforms

import networks
import utils

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
import pickle
import numpy as np

from PIL import Image

import re
from collections import OrderedDict

sys.path.append('./edm')

'''
pretrained model (currently VE-SDE on CIFAR10 no class conditional) from
https://github.com/NVlabs/edm
'''

from edm import dnnlib
from edm import torch_utils
from edm.torch_utils import *

import model_configs
import json

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost' #'115.145.136.28'
    os.environ['MASTER_PORT'] = '12355' #12356

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    dist.barrier()

def cleanup():
    dist.destroy_process_group()

def ema_update(model, ema_model, decay_rate):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data = decay_rate * ema_param.data + (1 - decay_rate) * param.data
    
# Main multi GPU training setup & loop
def train(rank, world_size, loss_fct, pretrained, timesteps, dataset, optimizer, lr_scheduler, model, ema_model, CHECKPOINT_PATH, hyperparams, tensorboard_path):
    
    setup(rank, world_size)
    
    if rank == 0:
        writer = SummaryWriter(tensorboard_path)
        
    model_obj = model.to(rank)
    ema_model_obj = ema_model.to(rank)
    #model = UNet2DModel.from_pretrained("cm_cifar10_pretrained").to(rank)
    #model.train()
    pretrained = pretrained.to(rank)
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=hyperparams['batch_size'], shuffle=False, drop_last=True, sampler=sampler, pin_memory=False, num_workers=0
    )
    
    model = DDP(model_obj, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    #pretrained = DDP(pretrained, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    ema_model = DDP(ema_model_obj, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    optim = optimizer(model.parameters(), lr=hyperparams['init_lr'])
    # Resume training #
    optim.load_state_dict(LOAD_PATH['optimizer_state_dict'])
    
    sigma_data = 0.5 #model.module.config['sigma_data']
    time_start_eps = 0.002 #model.module.config['time_start_epsilon']
    #sigma_data = model.config['sigma_data']
    #time_start_eps = model.config['time_start_epsilon']
    
    dist_dim = 3*(32**2) #3*(model.module.config['image_size']**2)
    #dist_dim = 3*(model.config['image_size']**2)
    stacked_cov_mats = torch.stack([torch.eye(dist_dim) for _ in range(hyperparams['batch_size'])], dim=0).to(rank)
    # setup dist to sample (B, C*H*W)
    multivariate_gaussian = MultivariateNormal(torch.zeros(hyperparams['batch_size'], dist_dim).to(rank), stacked_cov_mats)
    
    for epoch in range(1, hyperparams['epochs']+1):
        print('RANK', rank, 'EPOCH', epoch, flush=True)
        dataloader.sampler.set_epoch(epoch)
        
        #epoch_ecfd_loss = torch.tensor(0, dtype=torch.float).to(rank)
        epoch_pinn_loss = torch.tensor(0, dtype=torch.float).to(rank)
        epoch_bnd_cond_loss = torch.tensor(0, dtype=torch.float).to(rank)
        
        for i, (X, _) in enumerate(tqdm.tqdm(dataloader)):
            optim.zero_grad(set_to_none=True)
            
            #try:
            X_0 = X.to(rank)
        
            '''[1] Sample batch of data X_0 and their associated t_i's to calculate X_t_i = X_0 + t_i * Z'''
            t_i = torch.tensor(
                [timesteps[random.randint(1, len(timesteps)-1)] for _ in range(hyperparams['batch_size'])], dtype=torch.float, requires_grad=True
            ).view(-1, 1).to(rank)
            #Z = multivariate_gaussian.sample().view(-1, 3, model.module.config['sample_size'], model.module.config['sample_size']).to(rank)
            Z = multivariate_gaussian.sample().view(-1, 3, model.module.config['image_size'], model.module.config['image_size']).to(rank)
            X_t_i = X_0 + t_i.view(-1, 1, 1, 1) * Z # (B, C, H, W) + (B, 1, 1, 1) * (B, C, H, W)
            
            '''[2] Calculate A(X_t_i, t_i) w/ pretrained score network'''
            with torch.no_grad():
                A = torch.div(X_t_i - pretrained(X_t_i, t_i.squeeze(1)), t_i.view(-1, 1, 1, 1))
            
            '''[3] Forward thru model network K and form integral approximation k (k = c_skip*X_t + c_out*K, configured as in Consistency Models)'''
            '''repeating t for all pixel dims and averaging them inside model forward for one call to autograd to compute jacobian'''
            #t = t_i.view(-1, 1) 
            #t = torch.repeat_interleave(t, 3*A.size(-2)**2, dim=1)
            #t_i = t.mean(1)
            mod_out = model(t_i.squeeze(1), X_t_i)
            #mod_out = model(X_t_i, t_i)[0]

            # k = (sigma_data**2/((t_i - time_start_eps)**2 + sigma_data**2)).view(-1, 1, 1, 1)*X_t_i # c_skip*X_t
            # k += (sigma_data*(t_i - time_start_eps)/torch.sqrt(t_i**2 + sigma_data**2)).view(-1, 1, 1, 1)*mod_out # c_out*K

            # RESET PARAMETRIZATION OF K SUCH THAT X^_0 prediction has same form as Consistency Models #
            k = ((t_i - time_start_eps)**2/((t_i - time_start_eps)**2 + sigma_data**2)).view(-1, 1, 1, 1)*X_t_i
            k -= (sigma_data*(t_i - time_start_eps)/torch.sqrt(t_i**2 + sigma_data**2)).view(-1, 1, 1, 1)*mod_out
            ###########################################################################################

            '''
            FOLLOWING IS FOR TRYING vmap(jacfwd()):
            Configured simply as model approximating integral, rather than c_skip & c_out form => for convenience of grad computation wrt t
            => LATER CAN INCORPORATE c_skip & c_out directly in model definition in networks.py
            '''
            
            # def func(t_i): # Helper function to calculate jacobian of k wrt t_i
            #     Z = multivariate_gaussian.sample().view(-1, 3, model.module.config['image_size'], model.module.config['image_size']).to(rank)
            #     X_t_i = X_0 + t_i.view(-1, 1, 1, 1) * Z # (B, C, H, W) + (B, 1, 1, 1) * (B, C, H, W)
            #     A = torch.div(X_t_i - pretrained(X_t_i, t_i), t_i.view(-1, 1, 1, 1))
                
            #     k = (sigma_data**2/((t_i - time_start_eps)**2 + sigma_data**2)).view(-1, 1, 1, 1)*X_t_i # c_skip*X_t
            #     k += (sigma_data*(t_i - time_start_eps)/torch.sqrt(t_i**2 + sigma_data**2)).view(-1, 1, 1, 1)*model(X_t_i, t_i) # c_out*K
            #     k = k.view(hyperparams['batch_size'], -1) # (B, C*H*W)
            #     k_batch_list = [k[:, j] for j in range(k.size(1))] # List of batch t_i's for each element in C*H*W
            #     return tuple(k_batch_list)
        
            '''[4] Calculate network gradient wrt t & form PINN loss for backprop '''
            # k: (B, C, H, W), t_i: (B)

            '''
            Using torch.func vmap and jacfwd (jacobian forward) functionalities ... for vectorized batch-wise deriv. computation
            => getting autograd.Function usage must override setup_context staticmethod ERROR [even though Function not used]
            => retrying without vmap and using 'for' on batch dim -> STILL GIVING SAME ERROR
            => trying on simple dummy model -> STILL GIVING SAME ERROR -> SEEMS TO BE PARALLEL PROBLEM
            => with DDP find_unused_parameters=False & adding model output loss directly solves above BUT
                -> Cannot access data pointer of Tensor that doesn't have storage ERROR on second batch vmap(jacfwd()) calculation
            => removed vmap and used iteration on batch dim 
                -> still get same error [even for simple dummy model]
            => tried spawning each process manually and joining them
                -> still get same error (displayed per each process)
            => working without DDP on single process (GPU) works
            '''
            #t_i = t_i.unsqueeze(1)
            #print(t_i.size())
            #k_t_grad = vmap(jacfwd(model))(t_i, X_t_i) # Randomness due to dropout op.
            #k_t_grad = []
            # for b in range(X_0.size(0)):
            #     one_k_t_grad = jacfwd(model)(t_i[b], X_t_i[b, :, :, :])
            #     #one_k_t_grad = jacobian(model, create_graph=True)(t_i[b], X_t_i[b, :, :, :])
            #     k_t_grad.append(one_k_t_grad)
            #print(k_t_grad)
            #print(k_t_grad[0].size())
            #k_t_grad = k_t_grad.view(X_0.size(0), -1)
            #print(k.size())
            #k_t_grad = torch.cat(k_t_grad, 0)
            # print(k_t_grad.size())
            #k_t_grad = k_t_grad.view(X_0.size(0), -1)

            #print('HERE2', flush=True)
            #k_t_grad = torch.diagonal(torch.stack(jacobian(func, t_i), dim=2), 0, 0, 1).T
            #print('HERE3', flush=True)
            #k_t_grad = k_t_grad.view(-1, 3, model.module.config['image_size'], model.module.config['image_size'])
            #print('HERE4', flush=True)

            # backward for문 이용해 pixel별 편미분 계산
            # t_i = t_i.view(-1, 1)
            # k = k.view(k.size(0), -1)
            # t_grads = []
            # for p in range(k.size(1)):
            #     m = torch.zeros((k.size(0), k.size(1)), dtype=torch.float).to(rank)
            #     m[:, p] = 1
            #     t_i.grad = torch.zeros((k.size(0), 1), dtype=torch.float).to(rank)
            #     # if p == k.size(1)-1:
            #     #     k.backward(m)
            #     # else:
            #     k.backward(m, retain_graph=True)
            #     t_grads.append(t_i.grad)
            # k_t_grad = torch.cat(t_grads, 1).view(-1, A.size(1), A.size(2), A.size(3))
            
            # autograd 한번 계산으로 픽셀 편미분들 합만 loss에 사용
            # k_t_grad = grad(k, t_i, grad_outputs=torch.ones_like(k), create_graph=True, retain_graph=True, allow_unused=True)[0]
            # A = A.sum(3).sum(2).sum(1)            
            
            # autograd iteration으로 모든 픽셀별 편미분 계산
            # k = k.view(hyperparams['batch_size'], -1)
            # grads = []
            # for j in range(k.size(dim=1)):
            #     grads.append(grad(k[:, j], t_i, grad_outputs=torch.ones_like(k[:, j]), retain_graph=True)[0])
            #     print(j, flush=True)
            # k_t_grad = torch.stack(grads, dim=1)
            # k_t_grad = k_t_grad.view(-1, 3, model.module.config['image_size'], model.module.config['image_size'])
            
            #print(A.size(), k_t_grad.size())
            
            ''' GRAD COMPUTATION PROBLEM
            -k의 픽셀 당 t_i에 대한 편미분값 구해야 함
            -autograd은 픽셀 대한 편미분값들의 합만 한번에 구할 수 있음
            ->autograd 사용시 각 픽셀별 편미분 구하려면 iteration을 무려 3072(CIFAR10기준)번 돌아야해서 너무 비효율적
            -jacobian 사용시 따로 함수를 지정해줘야 하는데, 외부변수에 의존하게 되어 복잡도가 올라가고 프로그램이 무한루프/데드락에 걸린 거 같은 현상 발견
            '''

            ## CURRENT GRAD COMPUTATION USING AUTOGRAD (with repeating t and averaging trick) ##
            '''repeating t for all pixel dims and averaging them inside model forward for one call to autograd to compute jacobian'''
            k = k.view(k.size(0), -1)
            #k_t_grad = grad(k, t, grad_outputs=torch.ones_like(k), create_graph=True, retain_graph=True, allow_unused=True)[0]

            ##### !!!!!!!!!! CHANGED GRAD COMPUTATION FOR PINN LOSS !!!!!!!!!!!!!!!! #####
            def get_vjp(v):
                return torch.autograd.grad(k, t_i, v, retain_graph=True)
            #k_t_grad = (vmap(get_vjp)(torch.stack([torch.eye(k.size(1)).to(rank)]*k.size(0), 1)))[0]

            ### patch wise sum grads ###
            # 3072 row vectors (batch_size=1) takes up maximum memory
            # => there should be 3072 / batch_size patches (same number of row vectors)
            # => 1 patch = sum of batch_size pixels
            num_patches = k.size(1)//hyperparams['patch_size']
            sampler_matrix = torch.zeros((num_patches, k.size(1))).to(rank)
            for row in range(num_patches):
                sampler_matrix[row, hyperparams['patch_size']*row:hyperparams['patch_size']*(row+1)] = 1

            k_t_grad = (vmap(get_vjp)(torch.stack([sampler_matrix]*k.size(0), 1)))[0]
            #print(k_t_grad.element_size() * k_t_grad.nelement(), flush=True)
            #print(k_t_grad.size(), flush=True) #(num pixels, batch size, 1)
            k_t_grad = k_t_grad.squeeze(2).T
            # if rank == 0:
            #     print(k_t_grad.size(), flush=True)
            #     print(k_t_grad)

            ### for loop on pixel dim (in chunks) ###
            ## TO MATCH BATCH=1 SIZE: pixel_dim x pixel_dim = batch_size x pixel_dim x chunk_size (rows)
            ## num_iterations (num_chunks) = pixel_dim / chunk_size
            # k_t_grads = []
            # chunk_size = k.size(1) // k.size(0)
            # num_chunks = k.size(1) // chunk_size
            # id_matrix = torch.eye(k.size(1)).to(rank)
            # for chunk in range(num_chunks):
            #     cur_extractor_matrix = id_matrix[chunk*chunk_size:chunk_size*(chunk+1), :]
            #     batch_cur_extractor_matrix = torch.stack([cur_extractor_matrix]*k.size(0), 1)
            #     chunk_k_t_grad = (vmap(get_vjp)(batch_cur_extractor_matrix))[0]
            #     chunk_k_t_grad = chunk_k_t_grad.squeeze(2).T
            #     k_t_grads.append(chunk_k_t_grad)
            # k_t_grad = torch.cat(k_t_grads, 1)

            ## Using all pixels ##
            #A = A.view(X_0.size(0), -1)
            ## Using patches ##
            A = A.sum(-1).view(k.size(0), -1)
            #print(A.size())

            #ecfd_loss = utils.get_ECFD_Loss(X_0.view(X_0.size(0), -1), X_t_i.view(X_t_i.size(0), -1) - k, rank).to(rank)
            pinn_loss = (loss_fct(k_t_grad, A)).to(rank) # 1e-8*torch.abs(mod_out).mean().mean().mean()
            bnd_cond_loss = (loss_fct(k, X_t_i.view(X_t_i.size(0), -1) - X_0.view(X_0.size(0), -1))).to(rank)
            #epoch_ecfd_loss += ecfd_loss
            epoch_pinn_loss += pinn_loss
            epoch_bnd_cond_loss += bnd_cond_loss
            #loss = 0.1*ecfd_loss + pinn_loss
            loss = pinn_loss + bnd_cond_loss
            loss.backward()
            optim.step()

            model.zero_grad()
            
            print('Batch', i+1, 'Loss:', loss.item(), flush=True)
            #torch.cuda.empty_cache()
            # except Exception as e:
            #     print('EXCEPTION AT BATCH', i+1)
            #     #torch.cuda.empty_cache()
            #     continue

            ema_update(model, ema_model, 0.99)
            
        # Log Epoch Loss & Save Model
        torch.cuda.set_device(rank)

        epoch_pinn_loss /= i
        #epoch_ecfd_loss /= i
        epoch_bnd_cond_loss /= i
        
        dist.reduce(epoch_pinn_loss, dst=0)
        #dist.reduce(epoch_ecfd_loss, dst=0)
        dist.reduce(epoch_bnd_cond_loss, dst=0)
        
        if rank == 0:
            writer.add_scalar('PINN Loss', epoch_pinn_loss/world_size, epoch)
            #writer.add_scalar('ECFD Loss', epoch_ecfd_loss/world_size, epoch)
            writer.add_scalar('BND COND Loss', epoch_bnd_cond_loss/world_size, epoch)
            
            if epoch % hyperparams['save_freq'] == 0:
                torch.save(
                    {'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optim.state_dict(),
                    }, CHECKPOINT_PATH
                )
                torch.save(ema_model.state_dict(), CHECKPOINT_PATH[:-3] + '_EMA.pt')
    
    if rank == 0:
        writer.flush()
        writer.close()
        
    cleanup()

# For test: fitting on a single image
class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, transform=None, num_iterations=60000):
        self.image = Image.open(img_path)
        self.transform = transform
        self.num_iterations = num_iterations
    
    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        else:
            image = self.image
        
        return image, 0

    def __len__(self):
        return self.num_iterations


if __name__ == "__main__":
    
    print(torch.version.cuda)

    # a = torch.rand(3072, 1, 3072, dtype=torch.float)
    # print(sys.getsizeof(a))
    # print(a.element_size()*a.nelement())
    # quit()
    
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = False
    
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    #os.environ['WORLD_SIZE'] = '4'
    
    #print("???")
    #quit()
    
    # print(os.path.dirname(__file__))
    # quit()
    
    # Pre-trained EDM VE CIFAR10 UNCOND model loading test #
    #device = torch.device('cuda')
    #print(device)
    
    with open('edm-cifar10-32x32-uncond-ve.pkl', 'rb') as f:
        #print(f)
        pretrained = pickle.load(f)['ema']#.to(device)
  
    # print(pretrained)
    
    pretrained.eval()
    # input_dummy = torch.rand(1, 3, 224, 224).to(device)
    # out = pretrained(input_dummy, torch.tensor([2.0]).to(device))
    # print(out)
    
    #quit()
    
    # with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl') as f:
    #     net = pickle.load(f)['ema'].to(device)
        
    # print(net)
        
    # quit()
    ##########################################################
    
    # Preprocessing for cifar10
    # (1) scale to [0, 1) OR [-1, 1]
    # (2) combine train & test data
    # (3) random horizontal flip (may be used for augmentation)
    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # EDM choice [-1, 1]
            #transforms.Lambda(lambda x: (x+1)/2) - NCSN++ original choice [0, 1] 
        ]
    )
    
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=False, transform=transform) #transform=transform
    testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=False, transform=transform) #transform=transform
    
    #dataset = torch.utils.data.ConcatDataset([trainset, testset])

    ############ Experimenting with only first 1000 images due to CUDA memory issues with vmap #############
    trainset.data = trainset.data[:1000]
    trainset.targets = trainset.targets[:1000]
    dataset = trainset
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    # print(len(dataloader))
    # print(next(iter(dataloader))[0].shape)
    # quit()
    ########################################################################################################

    ############# Test: Fitting on one image ##################
    # image, _ = dataset[0]
    # image.save('./data/cifar10/sample_img.png')

    # dataset = SingleImageDataset('./data/cifar10/sample_img.png', transform=transform)

    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=512)
    # print(len(dataloader))
    # print(next(iter(dataloader))[0].shape)

    # quit()
    ###########################################################
    
    #augmented_dataset = transforms.RandomHorizontalFlip(p=0.5)(dataset)
    
    world_size = 6
    optimizer = RAdam # RectifiedAdam Optimizer (as used in Consistency Models)
    lr_scheduler = None #LambdaLR
    loss_fct = torch.nn.MSELoss()
    
    model = networks.NCSNpp(model_configs.NCSNpp_init_configs)
    #model = UNet2DModel.from_pretrained("cm_cifar10_pretrained")
    #model.train()
    #model = None

    #### CHECKING MODEL SIZE ####
    # param_size = 0
    # for param in model.parameters():
    #     param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()
    # print('MODEL SIZE:', (param_size + buffer_size) / 1024**2, 'MB')
    # quit()
    #############################
    
    model_name = 'cifar10_vmap_autograd_pinn_patchwise_ema_2nd'
    
    CHECKPOINT_PATH = './models/' + model_name + '.pt'
    tensorboard_path = './logs/' + model_name + '_train'

    # Resume model training (load model) #
    LOAD_PATH = './models/cifar10_vmap_autograd_pinn_patchwise_ema_1st.pt'
    model_dict = torch.load(LOAD_PATH['model_state_dict'])
    model_dict_ordered = OrderedDict()
    for k, v in model_dict.items():
        if re.search('module', k):
            new_key = re.sub(r'^module\.', '', k)
            model_dict_ordered[new_key] = v
        else:
            model_dict_ordered = model_dict
    model.load_state_dict(model_dict_ordered)

    # Resume model training (load ema_model) #
    ema_model = networks.NCSNpp(model_configs.NCSNpp_init_configs)
    LOAD_PATH = './models/cifar10_vmap_autograd_pinn_patchwise_ema_1st_EMA.pt'
    model_dict = torch.load(LOAD_PATH)
    model_dict_ordered = OrderedDict()
    for k, v in model_dict.items():
        if re.search('module', k):
            new_key = re.sub(r'^module\.', '', k)
            model_dict_ordered[new_key] = v
        else:
            model_dict_ordered = model_dict
    ema_model.load_state_dict(model_dict_ordered)
    
    hyperparams = {
        'epochs': 5000, # 800000 default choice in Consistency Models
        'batch_size': 50, # 512 default in Consistency Models, BUT CUDA_OUT_OF_MEMORY error
        'init_lr': 4e-4, # Default in Consistency Models, varied across different training schemes (4e-5)
        'save_freq': 1,
        'patch_size': 32
    }
    
    timesteps = utils.get_sigmas(model_configs.NCSNpp_init_configs) # currently 'edm_style' time discretization -> sigma_i = t_i for all i's
    
    start = time.time()

    # print("BUS ERROR???")
    # quit()
    
    mp.spawn(train, args=(
        world_size, loss_fct, pretrained, timesteps, dataset, optimizer, lr_scheduler, model, ema_model, CHECKPOINT_PATH, hyperparams, tensorboard_path
    ), nprocs=world_size, join=True)
    # mp.set_start_method('spawn')
    # children = []
    # for p in range(world_size):
    #     subproc = mp.Process(target=train, args=(p, world_size, loss_fct, pretrained, timesteps, dataset, optimizer, lr_scheduler, model, CHECKPOINT_PATH, hyperparams, tensorboard_path))
    #     children.append(subproc)
    #     subproc.start()
    # for p in range(world_size):
    #     children[p].join()

    total_training_time = time.time() - start
    
    print('\n TOTAL RUNNING TIME: \n', total_training_time)
    
    descriptions = {
        'hyperparameters': hyperparams,
        'model_name': 'NCSN++',
        'model_configurations': model_configs.NCSNpp_init_configs,
        'model_save_name': model_name,
        'pretrained_model': 'EDM_VE-SDE',
        'class_conditional': False,
        'training_time': total_training_time,
        'num_gpus': world_size,
        'lr_scheduler': None,
        'optimizer': 'RectifiedAdam',
        'dataset': 'CIFAR10',
        'extra_descriptions': 'data scaled to [-1, 1], train&test combined, used random horizontal flip with p=0.5 // weightings for loss: PINN = 1 & BND_COND = 1 // PINN loss grad calculated with vmap'
    }

    # descriptions = {
    #     'hyperparameters': hyperparams,
    #     'model_name': 'NCSN++_Reduced_for_CIFAR10',
    #     'model_configurations': 'As in ./cm_cifar10_pretrained/config.json',
    #     'model_save_name': model_name,
    #     'pretrained_model': 'EDM_VE-SDE',
    #     'class_conditional': False,
    #     'training_time': total_training_time,
    #     'num_gpus': world_size,
    #     'lr_scheduler': None,
    #     'optimizer': 'RectifiedAdam',
    #     'dataset': 'CIFAR10',
    #     'discretization_steps': 1000,
    #     'extra_descriptions': 'data scaled to [-1, 1], train&test combined, used random horizontal flip for data augmentation with p=0.5 // uniform weighting for pinn & ecfd loss // used pretrained CM NCSN++ Reduced for CIFAR-10 for weight init'
    # }
    
    with open('./models/descriptions/' + model_name + '.json', 'w') as fp:
        json.dump(descriptions, fp)