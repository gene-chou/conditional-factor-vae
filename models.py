import torch
import torch.nn as nn
import torch.nn.init as init 
#from dataloader import idx2onehot


class Discriminator(nn.Module):
    def __init__(self, latent_size, num_labels=10):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size
        self.net = nn.Sequential(
            nn.Linear(latent_size, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2),
            nn.Linear(1000, 2),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


# only concat labels when in 1D latent space 
# latent space dimension += number of labels 
class FactorVAE(nn.Module):

    def __init__(self, latent_size, num_labels=10):
        super(FactorVAE, self).__init__()
        self.latent_size = latent_size
        self.num_labels = num_labels
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), #in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(),
            nn.Conv2d(256, 2*latent_size, 1), #ouput from (3,224,192) is (10,11,9)

            #reshape latent space, kernel size has to be greater than 
            #1/2 of 11 and 9 for output to become (c,1,1)
            nn.AvgPool2d(6)
        )

        self.decode = nn.Sequential(
            nn.Conv2d((latent_size+num_labels), 256, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        #print(self._modules)
        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, c, no_decode=False):
        stats = self.encode(x)
        mu = stats[:, :self.latent_size]
        logvar = stats[:, self.latent_size:]
        z = self.reparameterize(mu, logvar)
        c = c.reshape(-1,self.num_labels,1,1)
        z_cat = torch.cat((z,c),dim=1)

        if no_decode:
            return z_cat.squeeze()
        else:
            #revert the pooling
            #print("z cat shape: {}".format(z_cat.shape))
            z_up = nn.functional.interpolate(z_cat, (11,9))

            recon_x = self.decode(z_up)
            return recon_x, mu, logvar, z_cat.squeeze()

    def inference(self, z, c):
        z = torch.cat((z,c),dim=-1) 
        #print("z1 shape: {}",z.shape)
        z = z.reshape(-1,self.num_labels+self.latent_size,1,1)
        #print("z2 shape: {}",z.shape)
        z = nn.functional.interpolate(z, (11,9))
        #print("z3 shape: {}",z.shape)
        recon_x = self.decode(z)
        return recon_x





#------------------MLP----------

class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, im_dim_mul,
                 conditional=True, onehot=False, num_labels=10):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.im_dim_mul = im_dim_mul

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, onehot, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, onehot, num_labels)

    def forward(self, x, c, no_dec=False):

        if x.dim() > 2:
            x = x.view(-1, self.im_dim_mul)

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)

        if no_dec:
            return z
        #z1 = torch.cat((z, c), dim=-1)
        recon_x = self.decoder(z, c)

        #print("z1 shape: ", z1.shape)

        return recon_x, means, log_var, z

    # proposed in original vae paper 
    # epsilon allows vae to be trained end by end because now mu+eps*std is a learnable parameter
    # very low value thereby not causing the network to shift away too much from the true distribution
    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, onehot, num_labels=10):

        super().__init__()

        self.conditional = conditional
        self.num_labels = num_labels
        if self.conditional:
            layer_sizes[0] += num_labels
        self.onehot = onehot
        self.MLP = nn.Sequential()

        #print("\n\nencoder loop: ", list(zip(layer_sizes[:-1], layer_sizes[1:])),"\n")
        # [(794, 256)] 

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1)
            # x shape becomes [64,794] from [64,784] (28x28 im -> 1x784 vector)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, onehot, num_labels=10):

        super().__init__()

        self.MLP = nn.Sequential()
        self.num_labels = num_labels
        self.conditional = conditional
        self.onehot = onehot
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size


        #print("\n\ndecoder loop: ", list(zip([input_size]+layer_sizes[:-1], layer_sizes)), "\n")
        #[(12, 256), (256, 784)] -> input and output shapes of decoder 

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

        #print(self.MLP)

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        recon_x = self.MLP(z)

        return recon_x
