import torch.nn as nn
import functools
import torch
import numpy as np
import net_layers
import utils
import math
import torch.nn.functional as F

#from diffusers import UNet2DModel

'''
NCSN++ model from (for VE-SDE & corresponding Reverse PF ODE -> currently no class conditional)
https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py

Our best score-based model for VE SDEs 1) uses FIR upsampling/downsampling, 2) rescales skip
connections, 3) employs BigGAN-type residual blocks, 4) uses 4 residual blocks per resolution
instead of 2, and 5) uses “residual” for input and no progressive growing architecture for output. - from ScoreSDE
'''

ResnetBlockBigGAN = net_layers.ResnetBlockBigGANpp
conv3x3 = net_layers.conv3x3
AttnBlockpp = net_layers.AttnBlockpp
Upsamplepp = net_layers.Upsamplepp
Downsamplepp = net_layers.Downsamplepp

'''
Current default configs (initial run)
1. no class conditional
2. positional time embedding only (no GaussianFourier for continuous training)
3. ResnetBlock from BigGAN
4. progressive sampling only for input w/ 'residual' style (during downsampling (pyramid), not for upsampling)
    -> can be changed to 'input_skip' type by setting config['resample'] = False
'''

# Simple dummy model for vmap, jacfwd parallel setting test
# class NCSNpp(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.fc1=nn.Linear(1, 3072) 
#         self.fc2=nn.Linear(3072, 20)
#         self.out=nn.Linear(20, 3072)

#         self.config = config

#     def forward(self, t, x):
#         #print(t.size(), x.size())
#         t = t.view(-1, 1)
#         x = x.view(-1, 3072)
#         #print(t.size(), x.size())
#         x=torch.tanh(self.fc1(t)) + x
#         x=torch.tanh(self.fc2(x))
#         x=self.out(x).view(-1, 3, 32, 32)
#         return x



class NCSNpp(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        # Get model configurations
        self.config = config
        #self.sigmas = utils.get_sigmas(config) -> currently 'edm_style' default discretization yields sigma_i = t_i for all i's
        self.time_embed_dim = config['time_embed_dim']
        self.num_res_blocks = config['num_res_blocks']
        self.attn_resolutions = config['attn_resolutions']
        self.dropout = config['dropout']
        self.channel_multiplier = config['channel_multiplier']
        self.num_resolutions = len(self.channel_multiplier)
        self.all_resolutions = [config['image_size'] // (2**i) for i in range(self.num_resolutions)]
        self.resample = config['resample']
        self.fir = config['sample_fir']
        self.fir_kernel = config['sample_fir_kernel']
        self.init_scale = config['init_scale']
        self.skip_rescale = config['skip_rescale']
        self.actv_fct = utils.get_act(config)
        
        # Setup AttnBlock, Upsample, Downsample, ResnetBlock
        AttnBlock = functools.partial(
            AttnBlockpp, init_scale=self.init_scale, skip_rescale=self.skip_rescale 
        )
        Upsample = functools.partial(
            Upsamplepp, with_conv=self.resample, fir=self.fir, fir_kernel=self.fir_kernel
        )
        Downsample = functools.partial(
            Downsamplepp, with_conv=self.resample, fir=self.fir, fir_kernel=self.fir_kernel
        )
        ResnetBlock = functools.partial(
            ResnetBlockBigGAN, act=self.actv_fct, dropout=self.dropout, fir=self.fir, 
            fir_kernel=self.fir_kernel, init_scale=self.init_scale, skip_rescale=self.skip_rescale, 
            temb_dim=self.time_embed_dim*4
        )
        
        ## Start Initializing Layers
        modules = []
        
        # Initialize Timestep Embedding Layers
        modules.append(nn.Linear(self.time_embed_dim, 4*self.time_embed_dim))
        #modules[-1].weight.data = net_layers.default_init()(modules[-1].weight.shape)
        #nn.init.zeros_(modules[-1].bias)
        modules.append(nn.Linear(4*self.time_embed_dim, 4*self.time_embed_dim))
        #modules[-1].weight.data = net_layers.default_init()(modules[-1].weight.shape)
        #nn.init.zeros_(modules[-1].bias)
        
        modules.append(conv3x3(3, self.time_embed_dim))
        input_pyramid_channel = 3
        
        # Initialize Downsampling Layers
        hidden_channels = [self.time_embed_dim]
        in_channel = self.time_embed_dim
        for res_lvl in range(self.num_resolutions):
            for block in range(self.num_res_blocks):
                out_channel = self.time_embed_dim * self.channel_multiplier[res_lvl]
                modules.append(ResnetBlock(in_ch=in_channel, out_ch=out_channel))
                in_channel = out_channel
                
                if self.all_resolutions[res_lvl] in self.attn_resolutions:
                    modules.append(AttnBlock(channels=in_channel))
                hidden_channels.append(in_channel)
                
            if res_lvl != self.num_resolutions-1:
                modules.append(ResnetBlock(down=True, in_ch=in_channel))
                modules.append(Downsample(in_ch=input_pyramid_channel, out_ch=in_channel))
                input_pyramid_channel = in_channel
                
                hidden_channels.append(in_channel)
                
        in_channel = hidden_channels[-1]
        modules.append(ResnetBlock(in_ch=in_channel))
        modules.append(AttnBlock(channels=in_channel))
        modules.append(ResnetBlock(in_ch=in_channel))
        
        # Initialize Upsampling Layers
        for res_lvl in reversed(range(self.num_resolutions)):
            for block in range(self.num_res_blocks+1):
                out_channel = self.time_embed_dim * self.channel_multiplier[res_lvl]
                modules.append(ResnetBlock(in_ch=in_channel + hidden_channels.pop(), out_ch=out_channel))
                in_channel = out_channel
                
            if self.all_resolutions[res_lvl] in self.attn_resolutions:
                modules.append(AttnBlock(channels=in_channel))
                
            if res_lvl != 0:
                modules.append(ResnetBlock(in_ch=in_channel, up=True))
        
        # Initialize Last Layers
        modules.append(nn.GroupNorm(num_groups=min(in_channel//4, 32), num_channels=in_channel, eps=1e-6))
        modules.append(conv3x3(in_channel, 3, init_scale=self.init_scale))
        self.all_modules = nn.ModuleList(modules)
        
    def forward(self, t, x):

        ## For vectorized batch-wise computation of jacobian of model wrt t
        # t = t.view(-1)
        # x = x.view(-1, 3, self.config['image_size'], self.config['image_size'])
        #print(t.size(), x.size())

        ## Averaging repeated t ##
        #t = t.mean(1)
        
        # Get timestep embeddings (positional)
        time_embedded = utils.get_timestep_embedding(t, self.time_embed_dim)
        
        # Forward thru Timestep Embedding Layers
        time_embedded = self.all_modules[0](time_embedded)
        time_embedded = self.all_modules[1](self.actv_fct(time_embedded))

        #print("NO PROBLEM TIME EMBEDDING LAYER")
        
        # Forward thru Downsampling Layers
        input_pyramid = x
        feature_maps = [self.all_modules[2](x)]
        mod_idx = 3
        
        for res_lvl in range(self.num_resolutions):
            for block in range(self.num_res_blocks):
                h = self.all_modules[mod_idx](feature_maps[-1], time_embedded)
                mod_idx += 1
                
                if h.shape[-1] in self.attn_resolutions:
                    h = self.all_modules[mod_idx](h)
                    mod_idx += 1
                    
                feature_maps.append(h)
                
            if res_lvl != self.num_resolutions - 1:
                h = self.all_modules[mod_idx](feature_maps[-1], time_embedded)
                mod_idx += 1
                
                input_pyramid = self.all_modules[mod_idx](input_pyramid)
                mod_idx += 1
                
                if self.skip_rescale:
                    input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                else:
                    input_pyramid = input_pyramid + h
                h = input_pyramid
                
                feature_maps.append(h)

        #print("NO PROBLEM DOWNSAMPLING LAYER")
            
        # Forward thru Layers btw Down&Upsampling
        h = feature_maps[-1]
        h = self.all_modules[mod_idx](h, time_embedded)
        mod_idx += 1
        h = self.all_modules[mod_idx](h)
        mod_idx += 1
        h = self.all_modules[mod_idx](h, time_embedded)
        mod_idx += 1

        #print("NO PROBLEM BTW LAYERS")
        
        # for j in range(len(feature_maps)):
        #     print(feature_maps[j].size())
        # print('------------------')
        
        # Forward thru Upsampling Layers
        for res_lvl in reversed(range(self.num_resolutions)):
            for block in range(self.num_res_blocks+1):
                #print(h.size(), feature_maps[-1].size())
                h = self.all_modules[mod_idx](torch.cat([h, feature_maps.pop()], dim=1), time_embedded)
                mod_idx += 1
                
            if h.shape[-1] in self.attn_resolutions:
                h = self.all_modules[mod_idx](h)
                mod_idx += 1
                
            if res_lvl != 0:
                h = self.all_modules[mod_idx](h, time_embedded)
                mod_idx += 1

        #print("NO PROBLEM UPSAMPLING LAYER")
        
        # Get Output
        h = self.actv_fct(self.all_modules[mod_idx](h))
        mod_idx += 1
        h = self.all_modules[mod_idx](h)
        mod_idx += 1

        #print("NO PROBLEM OUTPUT LAYER")
        
        return h #h # for has_aux option in jacfwd

    
if __name__ == "__main__":
    
    eps = 0.002
    rho = 7
    N = 1000#18
    T = 80
    
    for i in range(1, N+1):
        print((eps**(1/rho) + (i-1)/(N-1)*(T**(1/rho) - eps**(1/rho)))**rho)
    quit()

    '''    
    ## CUR DISCRETIZED t_i values:
    0.0020000000000000013
    0.0075280199627840725
    0.022934518372333384
    0.05994731123547159
    0.1395164687310165
    0.29644228447915727
    0.5853481231945422
    1.088170636545279
    1.9233398370400492
    3.256821519765537
    5.315194521796376
    8.400935309099825
    12.91008238075732
    19.352452980325207
    28.374584604156844
    40.78557379650796
    57.58598472124816
    80.0
    '''
    # import model_configs as configs
    
    device = torch.device('cuda')
    
    samp_x_ts = torch.rand(256, 3, 32, 32).to(device)
    samp_ts = torch.rand(256).to(device)
    # k_theta = NCSNpp(configs.NCSNpp_init_configs).to(device)
    # print(k_theta(samp_x_ts, samp_ts).size())

    # from model_configs import CM_U_Net_init_configs as config

    # k_theta = UNetModel(**config).to(device)
    # k_theta.load_state_dict('cd_imagenet64_lpips.pt')

    # k_theta =  UNet2DModel(
    #     sample_size=32,
    #     in_channels=3,
    #     out_channels=3,
    #     layers_per_block=4,
    #     attention_head_dim=8,
    #     block_out_channels=(128, 256, 256, 256),
    #     down_block_types=(
    #         "SkipDownBlock2D",
    #         "AttnSkipDownBlock2D",
    #         "SkipDownBlock2D",
    #         "SkipDownBlock2D",
    #     ),
    #     downsample_padding=1,
    #     act_fn="silu",
    #     center_input_sample=True,
    #     mid_block_scale_factor=math.sqrt(2),
    #     up_block_types=(
    #         "SkipUpBlock2D",
    #         "SkipUpBlock2D",
    #         "AttnSkipUpBlock2D",
    #         "SkipUpBlock2D",
    #     ),
    # ).to(device)

    import model_configs

    # k_theta = UNet2DModel(model_configs.CM_CIFAR10_init_configs)

    # k_theta.load_state_dict(torch.load('diffusion_pytorch_model.bin'))

    k_theta = UNet2DModel.from_pretrained("cm_cifar10_pretrained").to(device)
    # from diffusers import DiffusionPipeline

    # pipeline = DiffusionPipeline.from_pretrained(
    #     "consistency/cifar10-32-demo",
    #     custom_pipeline="consistency/pipeline",
    # )

    print(k_theta.parameters())
    print(k_theta(samp_x_ts, samp_ts)[0].size())
    