# Pretrained model
    VE SDE EDM from 
    'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl'

# View running job: 
    $ squeue -u x2452a12

# To start conda env:
    $ module load python/3.7.1
    $ source activate lsgm_pinn

# If permission denied when accessing tensorboard:
    $ export TMPDIR=/tmp/USER; 
    $ mkdir -p $TMPDIR;

# MODELS

    -cifar10_1000steps_0530.pt: used 1000 timestep discretizations => suddenly loss yielding NaN's (at approx 50th epoch)
    -cifar10_2000steps_0602.pt: used rounding to 10 digits for timesteps from above model => still yielding NaN's (at approx 80th epoch)
    -cifar10_2000steps_0603.pt: seems to be exploding gradients problem, so reduced lr to 4e-4 and cancelled rounding => still yielding NaN's (at approx 270th epoch)
    -cifar10_2000steps_0608.pt: switched discretization steps from 1000 back to 18 (as in consistency models originally) => still yielding NaN's (at approx 253th epoch)
    -cifar10_2000steps_0608-2.pt: reduced learning rate to 4e-5 (somehow stopped at 713th epoch - NaN's not appearing but for some reason was denied access to tensorboard logging directory in the middle and stopped...)
    -cifar10_2000steps_0609.pt: loss weightings changed [pinn = 1, ecfd = 0.1], switched discretization steps back to 1000
        => loss seems to be decreasing for ecfd, oscillating for pinn
    -cifar10_2000steps_0611_only_pinn.pt: only used pinn loss with lr 4e-4 (default in consistency models)
        => loss requires using model output explicitly so added reg term of mod output L1 norm (weighted by 1e-8)
    -cifar10_4000steps_0611_only_pinn.pt: trained for 2000 more steps after loading 'cifar10_2000steps_0611_only_pinn.pt'

    # After 0615 meeting #

    -cifar10_2000steps_0619_bnd_cond_added.pt: used pinn loss and boundary condition loss, reset parametrization of k network such that it is inline with consistency models (ours is X^_0 = X_t - k), time discretization 100 steps

    => might also consider using X_0 directly from sample instead of pretrained network (from which score and thus A is derived)
    
    -cifar10_4000steps_0619_bnd_cond_added.pt: trained for 2000 steps more for above model (with same configurations)

    ###### Fixed PINN loss computation to be accurate (pixel wise) ######
    -cifar10_vmap_autograd_pinn: fitting only on 1000 images due to memory constraint
    -cifar10_vmap_autograd_pinn_2nd_try: resumed training for above model with same configs

    # Patch-wise #
    -cifar10_vmap_autograd_pinn_patchwise: fitting on all 60000 CIFAR-10 images
    -cifar10_vmap_autograd_pinn_patchwise_ema_1st: fitting on 1000 images, using ema for update
    (_EMA: exponential weighted parameters version for generating images)
    -cifar10_vmap_autograd_pinn_patchwise_ema_2nd: resumed training, including optimizer, for 1st model