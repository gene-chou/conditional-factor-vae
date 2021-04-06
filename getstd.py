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

    def loss_fn(recon_x, x, mean, log_var):
        #print(x.view(-1, im_dim_multiply(im_dim)))
        #BCE = torch.nn.functional.binary_cross_entropy(
        #    recon_x.view(-1, im_dim_multiply(im_dim)), x.view(-1, im_dim_multiply(im_dim)), reduction='sum')
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, im_dim_multiply(im_dim)), x.view(-1, im_dim_multiply(im_dim)), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)

    vae = VAE(
        encoder_layer_sizes=encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=decoder_layer_sizes,
        im_dim_mul = im_dim_multiply(im_dim),
        conditional=args.conditional,
        onehot = False,
        num_labels=10).to(device)

    vae.load_state_dict(torch.load("results/baseline_cmlp_10epochs_l8.pth"))

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    # std of each dimension of all images 
    dim_sum = []
    for i in range(args.latent_size+args.num_labels):
        dim_sum.append([])

    dim_std = []


    # only 1 epoch for inference 
    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            recon_x, mean, log_var, z = vae(x, y)

            #print("z shape: ",z.shape, z[0].shape, z[0])
            #print("z type: ", type(z), type(z[0]))

            for item in z:
                for idx, dim in enumerate(item):
                    #print("append dim: ", dim)
                    dim_sum[idx].append(float(dim))

            #print(dim_sum)

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))

        #torch.save(vae.state_dict(), "results/baseline_cmlp_30epochs_l8.pth")

    import statistics 
    for dim in dim_sum: 
        dim_std.append(statistics.stdev(dim))

    print("z shape: ", z.shape, z[0].shape)
    print("dim sum list: ",len(dim_sum), len(dim_sum[0]), len(dim_sum[1]))
    print("dim_std list: ", len(dim_std), "\n", dim_std)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--print_every", type=int, default=1000)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", type=bool, default=True)
    parser.add_argument("--dataset", type=str, default='celeba')
    parser.add_argument("--num_labels", type=int, default=10)

    args = parser.parse_args()

    main(args)
