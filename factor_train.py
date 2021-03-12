import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import dataloader
#from dataloader import get_selected_celeba_dataset
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from collections import defaultdict

from models import FactorVAE, Discriminator


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'celeba_torchvision':
        dataset = torchvision.datasets.CelebA(
            root='data', split='train', target_type='attr', transform=transforms.ToTensor(),
            download=True)
    elif args.dataset == 'celeba':
        dataset = dataloader.get_celeba_selected_dataset()
      
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)


    def loss_fn(recon_x, x, mean, log_var):
        #BCE = torch.nn.functional.binary_cross_entropy(
        #    recon_x.view(-1, im_dim_multiply(im_dim)), x.view(-1, im_dim_multiply(im_dim)), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        BCE = binary_cross_entropy_with_logits(recon_x, x, reduction=args.bce_reduction)

        return (BCE + KLD) / x.size(0)

    def permute_dims(z):
    	assert z.dim() == 2

    	B, _ = z.size()
    	perm_z = []
    	for z_j in z.split(1, 1):
            perm = torch.randperm(B).to(z.device)
            perm_z_j = z_j[perm]
            perm_z.append(perm_z_j)
	
    	return torch.cat(perm_z, 1)

    vae = FactorVAE(
        latent_size=args.latent_size, num_labels=args.num_labels).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    D = Discriminator(args.latent_size).to(device)
    optim_D = torch.optim.Adam(D.parameters(), lr=args.learning_rate)
    zeros = torch.zeros(args.batch_size, dtype=torch.long, device=device)
    ones = torch.ones(args.batch_size, dtype=torch.long, device=device)

    logs = defaultdict(list)
    total_loss_log = []

    for epoch in range(args.epochs):
        
        if epoch%5==0:
            optimizer.param_groups[0]['lr'] /= 10
            optim_D.param_groups[0]['lr'] /= 10
            print("\nnew learning rate at epoch {} is {}!\n".format(epoch, optimizer.param_groups[0]['lr']))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            recon_x, mean, log_var, z = vae(x, y)
            loss = loss_fn(recon_x, x, mean, log_var)

            #print("z shape: ",z.shape)
            D_z = D(z)
            tc_loss = (D_z[:,:1] - D_z[:,1:]).mean()

            total_loss = loss+tc_loss*6.4 #gamma parameter 

            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)

            #---also train discriminator
            z_prime = vae(x, y, no_decode=True)
            z_prime = permute_dims(z_prime)
            D_z_prime = D(z_prime)
            D_tc_loss = 0.5*cross_entropy(D_z, zeros) + 0.5*cross_entropy(D_z_prime, ones)

            optim_D.zero_grad()
            D_tc_loss.backward()

            optimizer.step() #after D_tc_loss so no inplace error
            optim_D.step()

            logs['loss'].append(total_loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, total_loss.item()))

		#print one recon_x as well as inference 
                recon_im = torch.transpose(recon_x[0],0,2)
                recon_im = torch.transpose(recon_im,0,1)
                plt.imshow((recon_im.cpu().data.numpy()*255).astype(np.uint8))
                plt.axis('off')
                
                if not os.path.exists(os.path.join(args.fig_root, args.exp_name)):
                    if not(os.path.exists(os.path.join(args.fig_root))):
                        os.mkdir(os.path.join(args.fig_root))
                    os.mkdir(os.path.join(args.fig_root, args.exp_name))

                plt.savefig(
                    os.path.join(args.fig_root, args.exp_name,
                                 "E{:d}||{:d}_recon.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                
                c = torch.randint(low=0, high=2, size=(1,10)) #populated with 0s and 1s
                for i in range(9):
                    c = torch.cat((c, torch.randint(low=0, high=2, size=(1,10))), 0)
                c = c.to(device)
                z = torch.randn([c.size(0), args.latent_size]).to(device)
                x = vae.inference(z, c=c)

                plt.figure()
                plt.figure(figsize=(5, 10))
                for p in range(10):
                    plt.subplot(5, 2, p+1) #row, col, index starting from 1
                    plt.text(
                        0, 0, "c={}".format(c[p]), color='black',
                        backgroundcolor='white', fontsize=8)

                    # this order is required for visualizing data for fully connected; easy to mess up color channels
                    p = x[p].view(3,224,192) if len(x.shape)<4 else x[p]
                    image = torch.transpose(p,0,2)
                    image = torch.transpose(image,0,1)
                    plt.imshow((image.cpu().data.numpy()*255).astype(np.uint8))
                    plt.axis('off')        

                plt.savefig(
                    os.path.join(args.fig_root, args.exp_name,
                                 "E{:d}||{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')

        total_loss_log.append( (epoch, total_loss.item()))

    print(total_loss_log)


if __name__ == '__main__':

    ts = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--latent_size", type=int, default=10)
    parser.add_argument("--print_every", type=int, default=500)
    parser.add_argument("--fig_root", type=str, default='results')
    parser.add_argument("--exp_name", type=str, default='test_{}'.format(str(ts)))
    parser.add_argument("--dataset", type=str, default='celeba')
    parser.add_argument("--num_labels",type=int, default=10)
    parser.add_argument("--bce_reduction",type=str, default='sum')

    args = parser.parse_args()

    main(args)
