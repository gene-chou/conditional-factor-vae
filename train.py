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
    image_gather = DataGather('true', 'recon')

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
    mkdirs(ckpt_dir)
    start_epoch = 0
    if args.ckpt_load:
        load_checkpoint(pbar, ckpt_dir, D, vae, optim_D, optim_vae, lr_vae, lr_D)
        #optim_D.param_groups[0]['lr'] = 0.00001#lr_D
        #optim_vae.param_groups[0]['lr'] = 0.00001#lr_vae
        print("confirming lr after loading checkpoint: ", optim_vae.param_groups[0]['lr'])

    # Output
    output_dir = os.path.join(args.output_dir, args.name)
    mkdirs(output_dir)


    ones = torch.ones(args.batch_size, dtype=torch.long, device=device)
    zeros = torch.zeros(args.batch_size, dtype=torch.long, device=device)

    for epoch in range(start_epoch, args.epochs):
        pbar.update(1)
        
        for iteration, (x, y, x2, y2) in enumerate(data_loader):

            x, y, x2, y2 = x.to(device), y.to(device), x2.to(device), y2.to(device)

            recon_x, mean, log_var, z = vae(x, y)

            if z.shape[0]!=args.batch_size:
                print("passed a batch in epoch {}, iteration {}!".format(epoch, iteration))
                continue

            D_z = D(z)
            
            vae_recon_loss = recon_loss(x, recon_x) * args.recon_weight
            vae_kld = kl_divergence(mean, log_var)
            vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean() * args.gamma
            vae_loss = vae_recon_loss + vae_tc_loss #+ vae_kld

            optim_vae.zero_grad()
            vae_loss.backward(retain_graph=True)

            z_prime = vae(x2, y2, no_dec=True) 
            z_pperm = permute_dims(z_prime).detach()
            D_z_pperm = D(z_pperm)
            D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

            optim_D.zero_grad()
            D_tc_loss.backward()
            optim_vae.step()
            optim_D.step()

            if iteration%args.print_iter == 0:
                pbar.write('[epoch {}/{}, iter {}/{}] vae_recon_loss:{:.4f} vae_kld:{:.4f} vae_tc_loss:{:.4f} D_tc_loss:{:.4f}'.format(
                epoch, args.epochs, iteration, len(data_loader)-1, vae_recon_loss.item(), vae_kld.item(), vae_tc_loss.item(), D_tc_loss.item()))

            
            if iteration%args.output_iter == 0 and iteration!=0:
                output_dir = os.path.join(args.output_dir, args.name)#, "{}.{}".format(epoch, iteration))
                mkdirs(output_dir)

                #reconstruction 
                #image_gather.insert(true=x.data.cpu(), recon=torch.sigmoid(recon_x).data.cpu())
                #data = image_gather.data
                #true_image = data['true'][0]
                #recon_image = data['recon'][0]
                #true_image = make_grid(true_image)
                #recon_image = make_grid(recon_image)
                #sample = torch.stack([true_image, recon_image], dim=0)
                #save_image(tensor=sample.cpu(), fp=os.path.join(output_dir, "recon.jpg"))
                #image_gather.flush()

                #inference given num_labels = 10
                c = torch.randint(low=0, high=2, size=(1,10)) #populated with 0s and 1s
                for i in range(9):
                    c = torch.cat((c, torch.randint(low=0, high=2, size=(1,10))), 0)
                c = c.to(device)
                z_inf = torch.rand([c.size(0), args.z_dim]).to(device)
                #print("shapes: ",z_inf.shape, c.shape)
                #c = c.reshape(-1,args.num_labels,1,1)
                z_inf = torch.cat((z_inf,c),dim=1)
                z_inf = z_inf.reshape(-1, args.num_labels+args.z_dim, 1,1)
                x = vae.decode(z_inf)

                plt.figure()
                plt.figure(figsize=(10, 20))
                for p in range(args.num_labels):
                    plt.subplot(5, 2, p+1) #row, col, index starting from 1
                    plt.text(0, 0, "c={}".format(c[p]), color='black',
                                backgroundcolor='white', fontsize=10)

                    p = x[p].view(3,218,178)
                    image = torch.transpose(p,0,2)
                    image = torch.transpose(image,0,1)
                    plt.imshow((image.cpu().data.numpy()*255).astype(np.uint8))
                    plt.axis('off')

                plt.savefig(
                    os.path.join(output_dir, "E{:d}||{:d}.png".format(epoch, iteration)), dpi=300)
                plt.clf()
                plt.close('all')

        if epoch%8==0:
            optim_vae.param_groups[0]['lr'] /= 10
            optim_D.param_groups[0]['lr'] /= 10
            print("\nnew learning rate at epoch {} is {}!".format(epoch, optim_vae.param_groups[0]['lr']))

        if epoch%args.ckpt_iter_epoch == 0:
            save_checkpoint(pbar, epoch, D, vae, optim_D, optim_vae, ckpt_dir, epoch)

    pbar.write("[Training Finished]")
    pbar.close()


def save_checkpoint(pbar, epoch, D, vae, optim_D, optim_vae, ckpt_dir, ckptname, verbose=True):
    model_states = {'D':D.state_dict(),
                    'vae':vae.state_dict()}
    optim_states = {'optim_D':optim_D.state_dict(),
                    'optim_VAE':optim_vae.state_dict()}
    lr_states = {'lr_vae': optim_vae.param_groups[0]['lr'], 'lr_D': optim_D.param_groups[0]['lr']}
    states = {'epoch':epoch,
              'model_states':model_states,
              'optim_states':optim_states,
              'lr_states':lr_states}

    filepath = os.path.join(ckpt_dir, str(ckptname))
    with open(filepath, 'wb+') as f:
        torch.save(states, f)
    if verbose:
        pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, epoch))

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
    parser.add_argument('--name', type=str, default='sum_dim128_nokld')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument('--gamma', default=5, type=float)
    parser.add_argument('--recon_weight', type=float, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    
    parser.add_argument("--lr_vae", type=float, default=0.001)
    parser.add_argument("--lr_D", type=float, default=0.001)
    parser.add_argument("--num_labels", type=int, default=10)

    parser.add_argument("--print_iter", type=int, default=1000)
    parser.add_argument("--output_iter", type=int, default=1500)
    parser.add_argument("--ckpt_iter_epoch", type=int, default=5)
    parser.add_argument("--ckpt_load", type=bool, default=True)
    parser.add_argument("--ckpt_dir", type=str, default='ckpt_output')
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    main(args)
