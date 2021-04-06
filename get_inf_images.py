import os
import time
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST, CelebA
from torch.utils.data import DataLoader
from dataloader import get_celeba_selected_dataset, mnist_check
from collections import defaultdict

from models import VAE


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    dataset = get_celeba_selected_dataset()
    im_dim = (218, 178, 3)
    encoder_layer_sizes = [116412, 256]
    decoder_layer_sizes = [256, 116412]

    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)


    def im_dim_multiply(im_dim):
        dim = 1
        for i in im_dim:
            dim *= i    
        return dim 


    vae = VAE(
        encoder_layer_sizes=encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=decoder_layer_sizes,
        im_dim_mul = im_dim_multiply(im_dim),
        conditional=args.conditional,
        onehot = False,
        num_labels=10).to(device)

    vae.load_state_dict(torch.load("results/baseline_cmlp_15epochs_l8.pth"))
        

    for label in range(args.num_labels):

        c = torch.randint(low=0, high=2, size=(1,10)) #populated with 0s and 1s
        for i in range(99):
            c = torch.cat((c, torch.randint(low=0, high=2, size=(1,10))), 0)

        for i in range(c.shape[0]):
            c[i][label] = 1 # 1? or 0.5? 0?

        print("c: ",c)

        c = c.to(device)
        z = torch.randn([c.size(0), args.latent_size]).to(device)

        print("z shape: ",z.shape)

        # this is after generating the images and getting the latent once more 
        # for i in range(z.shape[0]):
        #     z[i][0] /= 6.20011844682385
        #     z[i][1] /= 4.1008276413666875

        #for i in range(z.shape[0]):
        #    z[i][args.latent_size] = 1 

        x = vae.inference(z, c=c)

        print("x shape: ",x.shape)


        if not os.path.exists(os.path.join(args.fig_root, "l8_100images")):
            if not(os.path.exists(os.path.join(args.fig_root))):
                os.mkdir(os.path.join(args.fig_root))
            os.mkdir(os.path.join(args.fig_root, "l8_100images"))


        plt.figure()
        
        for img_idx, p in enumerate(x):
            p = p.view(3,218,178)
            image = torch.transpose(p,0,2)
            image = torch.transpose(image,0,1)
            plt.imshow(image.cpu().data.numpy())

            plt.axis('off')

        
            plt.savefig(
                os.path.join(args.fig_root, "l8_100images",
                             "l{}_{}.png".format(label, img_idx)),
                dpi=300)
            plt.clf()
        
        plt.close('all')

        break



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--print_every", type=int, default=500)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", type=bool, default=True)
    parser.add_argument("--dataset", type=str, default='celeba')
    parser.add_argument("--num_labels", type=int, default=10)

    args = parser.parse_args()

    main(args)
