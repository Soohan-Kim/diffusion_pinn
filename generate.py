import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision.transforms import ToPILImage, ToTensor, Resize
import torchvision.transforms as transforms
import torchvision
import model_configs
import re
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler
import networks
import PIL
#from diffusers import UNet2DModel

# Function to convert tensor to numpy array and resize images
def postprocess_images(images):
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    images = (images + 1) / 2.0  # Scale from [-1, 1] to [0, 1]
    images = images * 255.0  # Scale from [0, 1] to [0, 255]
    images = images.astype(np.uint8)
    return images

# Function to calculate the FID score
def calculate_fid_score(real_images, generated_images):
    # Load pre-trained InceptionV3 model
    inception_model = inception_v3(pretrained=True, transform_input=False).eval().to('cuda')
    
    # Resize images to match InceptionV3 requirements (299x299 pixels)
    resize_transform = transforms.Compose([ToTensor(), Resize((299, 299))])
    real_images_resized = torch.stack([resize_transform(ToPILImage()(image)) for image in real_images], dim=0).float().numpy()
    generated_images_resized = torch.stack([resize_transform(ToPILImage()(image)) for image in generated_images], dim=0).float().numpy()
    
    # Convert resized images to PyTorch tensors
    real_images_tensor = torch.tensor(real_images_resized).float().to('cuda') #* 255.0
    generated_images_tensor = torch.tensor(generated_images_resized).float().to('cuda') #* 255.0
    
    # Extract features from the pool3 layer of the InceptionV3 model
    # real_features = inception_model(real_images_tensor)#[0].view(real_images_tensor.shape[0], -1).detach().cpu().numpy()
    # print(real_features.size())
    # quit()
    # generated_features = inception_model(generated_images_tensor)[0].view(generated_images_tensor.shape[0], -1).detach().cpu().numpy()
    real_features = inception_model(real_images_tensor).detach().cpu().numpy()
    generated_features = inception_model(generated_images_tensor).detach().cpu().numpy()
    
    # Calculate mean and covariance of real and generated image features
    mu_real = np.mean(real_features, axis=0)
    mu_generated = np.mean(generated_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_generated = np.cov(generated_features, rowvar=False)
    
    # Calculate the squared root of the product of covariance matrices
    cov_sqrt = sqrtm(sigma_real @ sigma_generated)
    
    # Calculate the FID score
    fid_score = np.linalg.norm(mu_real - mu_generated)**2 + np.trace(sigma_real + sigma_generated - 2 * cov_sqrt)
    
    return fid_score

def load_model_n_generate_imgs(model_path, num_imgs, configs):
    device = torch.device('cuda')

    # Load trained model
    model = networks.NCSNpp(configs).to(device)
    #model = UNet2DModel.from_pretrained("cm_cifar10_pretrained").to(device)
    model_dict = torch.load(model_path)

    model_dict_ordered = OrderedDict()
    #pattern = re.compile('module.')
    #pattern = re.compile(r'^module\.')

    for k, v in model_dict.items():
        if re.search('module', k):
            new_key = re.sub(r'^module\.', '', k)
            #new_key = re.sub(pattern, ', k')
            #model_dict_ordered[re.sub(pattern, '', k)] = v
            model_dict_ordered[new_key] = v
        else:
            model_dict_ordered = model_dict

    model.load_state_dict(model_dict_ordered)

    # Generate images
    timestep = configs['time_end_T']
    time_start_eps = configs['time_start_epsilon']
    sigma_data = configs['sigma_data']
    model.eval()
    with torch.no_grad():
        latents = torch.randn([num_imgs, 3, configs['image_size'], configs['image_size']], dtype=torch.float, device=device) * timestep
        #latents = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(latents) * timestep
        #t = torch.tensor([timestep]*num_imgs, dtype=torch.float, device=device)
        t = torch.tensor([timestep]*num_imgs, dtype=torch.float, device=device)
        #t_in = t.view(-1, 1)
        #t_in = torch.repeat_interleave(t_in, 3*configs['image_size']**2, dim=1)

        mod_out = model(t, latents) #model(latents, t)[0] 

        k = ((t - time_start_eps)**2/((t - time_start_eps)**2 + sigma_data**2)).view(-1, 1, 1, 1)*latents # c_skip*X_T
        k -= (sigma_data*(t - time_start_eps)/torch.sqrt(t**2 + sigma_data**2)).view(-1, 1, 1, 1)*mod_out # c_out*K
        gen_imgs = latents-k #k

    return gen_imgs

def get_real_imgs(num_imgs):
    tfs = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=False, transform=tfs)
    random_sampler = RandomSampler(dataset, replacement=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=random_sampler)
    real_imgs = []
    for img in dataloader:
        real_imgs.append(img[0])

        if len(real_imgs) >= num_imgs:
            break

    real_imgs = torch.cat(real_imgs, dim=0)

    return real_imgs

if __name__ == "__main__":
    path_name = 'cifar10_vmap_autograd_pinn_patchwise_ema_1st_EMA'

    batch_size = 512
    gen_imgs = load_model_n_generate_imgs('./models/' + path_name + '.pt', batch_size, model_configs.NCSNpp_init_configs)

    real_imgs = get_real_imgs(batch_size)

    # Postprocess generated images
    gen_imgs = postprocess_images(gen_imgs) # From [-1, 1] to [0, 255]
    real_imgs = postprocess_images(real_imgs)

    # Save generated images ...
    for i in range(gen_imgs.shape[0]):
        PIL.Image.fromarray(gen_imgs[i, :, :, :], 'RGB').save('./results/' + path_name + '/generated_image' + str(i) + '.png')

    # Consistency Model w/ pretrained score NFE 1 BEST CIFAR10 FID = 3.55
    print(calculate_fid_score(real_imgs, gen_imgs))