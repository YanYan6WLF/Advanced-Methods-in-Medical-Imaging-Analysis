__author__ = "Robin Sandkuehler"
__copyright__ = "Center for medical Image Analysis and Navigation, University of Basel, 2020"
__email__ = "robin.sandkuehler@unibas.ch"

import os
import torchvision as tv
import torch as th
from torch.utils import data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def KLLoss(mu, logvar):
    # enter the definition of the KL divergence if p(z|x) is Gaussian and p(z)= N(0, I)
    sigma_squared = th.exp(logvar)
    kl_divergence = 0.5 * th.sum(sigma_squared + mu**2 - 1 - logvar)
    return kl_divergence


class VAE(th.nn.Module):
    def __init__(self, kernel_size=3,in_channels=1,latent_dim=64):
        super().__init__()
        #define the architecture of the VAE

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=kernel_size, stride=2, padding=1),  # output: [32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=2, padding=1),  # output: [64, 7, 7]
            nn.ReLU(),
            nn.Flatten()  # output: [64*7*7]
        )
        self.fc_mu = nn.Linear(64*8*8, latent_dim)
        self.fc_logvar = nn.Linear(64*8*8, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 64*8*8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),  # output: [32, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),  # output: [1, 28, 28]
            nn.Sigmoid()  
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5*logvar)
        eps = th.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 64, 8, 8)
        return self.decoder(z)

    def forward(self, x):
     # implement forward pass
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

  



def train_mnist(result_path, num_epoch=10, batch_size=128, seed=123456):
    th.manual_seed(seed)
    np.random.seed(seed)

    device = th.device("cpu")

    # visualization
    summary_writer = SummaryWriter()

    # mdoel generatation
    model = VAE(kernel_size=3)
    model.to(device)
    model.train()

    # print number of model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("number of parameters model", params)

    # get MNIST data set
    transform = transforms.Compose(
        # If it works well with 32x32 try 64x64!
        [transforms.Resize((32, 32)),
         transforms.ToTensor()])

    data_set = tv.datasets.MNIST("/tmp/MNIST", train=True, transform=transform, target_transform=None, download=True)
    batch_size = batch_size
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 10}

    data_loader = data.DataLoader(data_set, **params)

    # define optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)

    ################################################################################################################
    ################################################################################################################
    # define the correct loss function
    reconstruction_loss = th.nn.MSELoss(reduction='sum')
    ################################################################################################################
    ################################################################################################################

    idx = 0
    loss_view = None
    for epoche in range(num_epoch):
        for image, target in data_loader:
            image = image.to(device)

            reconstructed_image, mu, sigma = model(image)

            recu_loss = reconstruction_loss(reconstructed_image, image)
            KL_loss = KLLoss(mu, sigma)

            loss = recu_loss + KL_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                opts = dict(title='input image', width=500, height=500)
                summary_writer.add_image('input_output',
                        th.cat([image[0, 0, ...].detach(),reconstructed_image[0, 0, ...]], dim=1),
                        dataformats='HW',
                        global_step=idx,
                        )

                state = {"epoche": epoche + 1, "model_state_dict": model.state_dict(),
                         "optimizer_state": optimizer.state_dict()}
                th.save(state, os.path.join(result_path, "model_minst_AE_" + str(idx) + ".pth"))

            summary_writer.add_scalar("reconstruction_loss", recu_loss.item(), global_step=idx)
            summary_writer.add_scalar("KL_loss", KL_loss.item(), global_step=idx)

            idx = idx + 1

            if idx % 250 == 0:
                latent_sample = th.randn(1, 64).to(device)
                genarated_image = model.decode(latent_sample).cpu()

                summary_writer.add_image('generated_image',
                        genarated_image[0, 0, ...].detach(),
                        dataformats='HW',
                        global_step=idx,
                        )

            print("Epoch", epoche, " Index", idx, "recu loss", recu_loss.item(), "KL loss", KL_loss.item())
    summary_writer.add_hparams(
            hparam_dict=dict(lr=0.001, num_epoch=num_epoch, batch_size=batch_size),
            metric_dict=dict(final_recon_loss=recu_loss.item(),
                final_kl_loss=KL_loss.item()),
            run_name='.'
            )


if __name__ == "__main__":
    train_mnist(result_path="/tmp")
    