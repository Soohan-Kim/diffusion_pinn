NCSNpp_init_configs = {
    'sigma_data': 0.5,
    'time_start_epsilon': 0.002,
    'time_end_T': 80,
    'discretization_steps': 100,
    'discretization_rho': 7,
    'discretization': 'edm_style',
    'time_embed_dim': 128,
    'num_res_blocks': 4,
    'attn_resolutions': (16,),
    'dropout': 0.1,
    'channel_multiplier': (1, 2, 2, 2),
    'image_size': 32,
    'resample': True,
    'sample_fir': False,
    'sample_fir_kernel': [1, 3, 3, 1],
    'skip_rescale': True,
    'init_scale': 0.0,
    'activation_func': 'swish'
}

CM_U_Net_init_configs = {
    'image_size': 32,
    'in_channels': 3,
    'model_channels': 128,
    'out_channels': 3,
    'num_res_blocks': 4,
    'attention_resolutions': (16,),
    'dropout': 0.1,
    'channel_mult': (1, 2, 2, 2),
    'conv_resample': True,
    'dims': 2,
    'num_classes': None,
    'use_checkpoint': True,
    'use_fp16': False,
    'num_heads': 1,
    'num_head_channels': -1,
    'num_heads_upsample': -1,
    'use_scale_shift_norm': False,
    'resblock_updown': False,
    'use_new_attention_order': False
}

CM_CIFAR10_init_configs = {
    "_class_name": "UNet2DModel",
    "_diffusers_version": "0.14.0",
    "act_fn": "silu",
    "add_attention": True,
    "attention_head_dim": 8,
    "block_out_channels": [
        128,
        256,
        256,
        256
    ],
    "center_input_sample": True,
    "class_embed_type": None,
    "down_block_types": [
        "SkipDownBlock2D",
        "AttnSkipDownBlock2D",
        "SkipDownBlock2D",
        "SkipDownBlock2D"
    ],
    "downsample_padding": 1,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 3,
    "layers_per_block": 1,
    "mid_block_scale_factor": 1.4142135623730951,
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "num_class_embeds": None,
    "out_channels": 3,
    "resnet_time_scale_shift": "default",
    "sample_size": 32,
    "time_embedding_type": "positional",
    "up_block_types": [
        "SkipUpBlock2D",
        "SkipUpBlock2D",
        "AttnSkipUpBlock2D",
        "SkipUpBlock2D"
    ]
}