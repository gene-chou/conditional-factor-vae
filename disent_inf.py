import os
from tqdm import tqdm
import torch
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from dataloader import get_celeba_selected_dataset
import numpy as np

from model import CelebaFactorVAE, Discriminator
from ops import recon_loss, kl_divergence, permute_dims
from utils import DataGather, mkdirs


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pbar = tqdm(total=args.epochs)

    dataset = get_celeba_selected_dataset()
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    lr_vae = args.lr_vae
    lr_D = args.lr_D
    vae = CelebaFactorVAE(args.z_dim, args.num_labels).to(device)
    optim_vae = torch.optim.Adam(vae.parameters(), lr=args.lr_vae)
    D = Discriminator(args.z_dim, args.num_labels).to(device)
    optim_D = torch.optim.Adam(D.parameters(), lr=args.lr_D, betas=(0.5, 0.9))

    # Checkpoint
    ckpt_dir = os.path.join(args.ckpt_dir, args.name)
    start_epoch = 0
    if args.ckpt_load:
        load_checkpoint(pbar, ckpt_dir, D, vae, optim_D, optim_vae, lr_vae, lr_D)
        print("confirming lr after loading checkpoint: ", optim_vae.param_groups[0]['lr'])

    # std of each dimension of all images 
    dim_sum = []
    for i in range(args.z_dim):#+args.num_labels):
        dim_sum.append([])

    dim_std = []

    for epoch in range(args.epochs):
        
        for iteration, (x, y, x2, y2) in enumerate(data_loader):

            x, y, x2, y2 = x.to(device), y.to(device), x2.to(device), y2.to(device)

            recon_x, mean, log_var, z = vae(x, y)

            for item in z:
                for idx, dim in enumerate(item):
                    #print("append dim: ", dim)
                    dim_sum[idx].append(float(dim))



    import statistics 
    for dim in dim_sum: 
        dim_std.append(statistics.stdev(dim))

    print("dim_std list: ", len(dim_std), "\n", dim_std, "\n")

    new_images = []
    new_latent = []
    norm_latent = []
    variance_all = []
    norm_latent_all = []
    var_correct = []


    for latent_idx in range(args.z_dim):

        c = torch.randint(low=0, high=2, size=(1,10)) #populated with 0s and 1s
        for i in range(19):
            c = torch.cat((c, torch.randint(low=0, high=2, size=(1,10))), 0)

        # for i in range(c.shape[0]):
        #     c[i][label] = 1 # fixing item in c <-- likely incorrect

        c = c.to(device)
        z = torch.rand([c.size(0), args.z_dim]).to(device)

        #print("z shape: ",z.shape)

        for i in range(z.shape[0]):
           z[i][latent_idx] = 1  # 1 is similar to avg value, if change to more extreme value, might have better results?

        #print("at latent idx {}, z: {}".format(latent_idx, z))

        z_cat = torch.cat((z,c),dim=1)
        z_cat = z_cat.reshape(-1,z_cat.shape[1],1,1)
        #print(z_cat.shape)
        x = vae.decode(z_cat) # returns reconstructed image 

        #print("x shape: ",x.shape) --> torch.Size([100, 116412])

        new_images.append(x)

        new_recon_x, new_mean, new_log_var, new_z = vae(x, c)

        new_latent.append(new_z)

        norm_latent = []

        #print("new_z shape: ",new_z.shape) # [100,8]
        for titem in new_z:
            item = titem.detach().cpu().numpy()
            #print("item shape: ",item.shape)
            norm_latent.append( np.divide(item, dim_std) )

        norm_latent = np.array(norm_latent)
        #print("norm_latent shape: ", norm_latent.shape) #[100,8]

#----------------

        min_var = 999
        variance = []
        for i in range(args.z_dim):
            var = statistics.variance(norm_latent[:, i])
            if var < min_var:
                min_var = var 
                min_idx = i
            variance.append(var)

        if min_idx == latent_idx:
            var_correct.append(1)
        else:
            var_correct.append(0)

        variance_all.append(variance)
        norm_latent_all.append(norm_latent)


    print("var correct: ", var_correct)
    print("variance: ", variance_all)

def load_checkpoint(pbar, ckpt_dir, D, vae, optim_D, optim_vae, lr_vae, lr_D, ckptname='last', verbose=True):
    if ckptname == 'last':
        #global ckpt_dir
        ckpts = os.listdir(ckpt_dir)
        if not ckpts:
            if verbose:
                pbar.write("=> no checkpoint found")
            return

        ckpts = [int(ckpt) for ckpt in ckpts]
        ckpts.sort(reverse=True)
        ckptname = str(ckpts[0])

    filepath = os.path.join(ckpt_dir, ckptname)
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            checkpoint = torch.load(f)

        epoch = checkpoint['epoch']
        vae.load_state_dict(checkpoint['model_states']['vae'])
        D.load_state_dict(checkpoint['model_states']['D'])
        optim_vae.load_state_dict(checkpoint['optim_states']['optim_VAE'])
        optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])
        #lr_vae.load_state_dict(checkpoint['lr_states']['lr_vae'])
        #lr_D.load_state_dict(checkpoint['lr_states']['lr_D'])
        pbar.update(epoch)
        if verbose:
            pbar.write("=> loaded checkpoint '{} (epoch {})'".format(filepath, epoch))
        global start_epoch 
        start_epoch = epoch
    else:
        if verbose:
            pbar.write("=> no checkpoint found at '{}'".format(filepath))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sum_dim128_1_6.4_batch512')
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr_vae", type=float, default=0.01)
    parser.add_argument("--lr_D", type=float, default=0.01)
    parser.add_argument('--gamma', default=5, type=float)
    parser.add_argument('--recon_weight', type=float, default=20)

    parser.add_argument("--num_labels", type=int, default=10)

    parser.add_argument("--print_iter", type=int, default=250)
    parser.add_argument("--output_iter", type=int, default=1000)
    parser.add_argument("--ckpt_iter_epoch", type=int, default=5)
    parser.add_argument("--ckpt_load", type=bool, default=True)
    parser.add_argument("--ckpt_dir", type=str, default='ckpt_output')
    parser.add_argument("--output_dir", type=str, default='output')

    args = parser.parse_args()

    main(args)
