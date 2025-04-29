__author__      = "Robin Sandkuehler"
__copyright__   = "Center for medical Image Analysis and Navigation, University of Basel, 2020"
__email__       = "robin.sandkuehler@unibas.ch"

import os
from datetime import datetime
import torchvision as tv
import torch as th
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter


class AE(th.nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

      #TODO: Implement your undercomplete autoencoder here
         # Encoder
        self.encoder = th.nn.Sequential(
            th.nn.Conv2d(1, 16, kernel_size, stride=1, padding=1),
            th.nn.ReLU(),
            th.nn.Conv2d(16, 32, kernel_size, stride=1, padding=1),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, kernel_size, stride=1, padding=1),
            th.nn.ReLU(),
            th.nn.MaxPool2d(2, stride=2)
        )
        
        # Decoder
        self.decoder = th.nn.Sequential(
            th.nn.Upsample(scale_factor=2, mode='nearest'),
            th.nn.ConvTranspose2d(64, 32, kernel_size, stride=1, padding=1),
            th.nn.ReLU(),
            th.nn.ConvTranspose2d(32, 16, kernel_size, stride=1, padding=1),
            th.nn.ReLU(),
            th.nn.ConvTranspose2d(16, 1, kernel_size, stride=1, padding=1),
            th.nn.Sigmoid() # To restrict output between 0 and 1
        )

    def forward(self, input):
        # TODO: the forward pass to compute the reconstructed image here
        x=self.encoder(input)
        reconstructed_image=self.decoder(x)
        return reconstructed_image

def add_noise(image):
    # TODO: add random noise to your input images to corrupt them
    noise = th.randn_like(image) * 0.2  # 
    noisy_image = image + noise  # 
    noisy_image = th.clamp(noisy_image, 0., 1.)  # 
    return noisy_image

def train_mnist(result_path, num_epoch=3, batch_size=100, seed=123456):

    th.manual_seed(seed)
    np.random.seed(seed)

    current_date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    result_path = os.path.join(result_path, current_date)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')  #put your model on the GPU if it is available

    # visualization
    summary_writer = SummaryWriter()

    # model generation
    model = AE(kernel_size=3)
    model.to(device)
    model.train()

    # print number of model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("number of parameters model", params)

    # get MNIST data set
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()])

    data_set_train = tv.datasets.MNIST("/tmp/MNIST", train=True, transform=transform, target_transform=None, download=True)
    data_set_test = tv.datasets.MNIST("/tmp/MNIST", train=False, transform=transform, target_transform=None, download=True)

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 4}

    data_loader_train = data.DataLoader(data_set_train, **params)   #training data loader
    data_loader_test = data.DataLoader(data_set_test, **params)    #test data loader

    # define optimizer
    lr=0.001
    optimizer = th.optim.Adam(model.parameters(), lr=0.001)#TODO define the optimizer

    loss = th.nn.MSELoss()#TODO define loss function

    idx = 0
    loss_view = None
    for epoche in range(num_epoch):
        print(epoche)
        for image, target in data_loader_train:
            image = image.to(device)
            #print(image.size())
	        #TODO: add noise to the images
            noisy_image=add_noise(image)
            #print(image.size())
            # TODO: compute the reconstructed images
            output=model(noisy_image)
            #TODO: compute the loss
            loss_value=loss(output,image)


            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            if idx % 50 == 0:
                summary_writer.add_image('input_output_train',
                        th.cat([image[0, 0, ...].detach(),output[0, 0, ...].detach()], dim=1),
                        global_step=idx,  # <-------------------------------- was missing in exercise - enables scrolling through time in tensorboard
                        dataformats='HW')

                state = {"epoche": epoche + 1, "model_state_dict": model.state_dict(),
                         "optimizer_state": optimizer.state_dict()}
                th.save(state, os.path.join(result_path, "model_minst_AE_" + str(idx) + ".pth"))

            summary_writer.add_scalar("loss", loss_value.item(), global_step=idx)
            idx = idx + 1

        print("Epoch", epoche, " Index", idx, "loss", loss_value.item())

    summary_writer.add_hparams(
            hparam_dict=dict(lr=lr, num_epoch=num_epoch, batch_size=batch_size),
            metric_dict=dict(final_loss=loss_value.item()),
            run_name='.'
            )
    
    model.eval()
    with th.no_grad():# 
     i=0
     while i<5:  #report 5 output test images
      for image, target in data_loader_test:
        i+=1

        image = image.to(device)

        noisy_image =add_noise(image)  # TODO: add noise to the input image

        reconstructed_image = model(noisy_image)# TODO: compute the reconstruction



        summary_writer.add_image('input_output_test',
                                     th.cat([image[0, 0, ...].detach(), noisy_image[0, 0, ...].detach(),reconstructed_image[0, 0, ...].detach()], dim=1),
                                     global_step=i,
                                     # <-------------------------------- was missing in exercise - enables scrolling through time in tensorboard
                                     dataformats='HW')




if __name__ == "__main__":
    train_mnist(result_path="/tmp")
