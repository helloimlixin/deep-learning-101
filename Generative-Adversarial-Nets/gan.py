#
# Created on Wed Jul 05 2023
#
# Copyright (c) 2023 Xin Li
# Email: helloimlixin@gmail.com
# All rights reserved.
#

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1), # leaky ReLU is usually good for GANs
            nn.Linear(128, 1), # discriminator outputs a scalar
            nn.Sigmoid() # GANs usually use sigmoid in the output layer
        )
    
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim), # 28x28x1 -> 1
            nn.Tanh() # normalize the inputs to [-1, 1] for tanh
        )
    
    def forward(self, x):
        return self.gen(x)

'''
Hyperparameters (GANs are very sensitive to hyperparameters)
'''
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4 # 3e-4 is the best learning rate for Adam, hands down -- Andrej Karpathy
z_dim = 64
image_dim = 28 * 28 * 1 # 784
batch_size = 32
num_epochs = 100

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss() # similar to the form of the GAN loss
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader): # we are not using labels, GANs are unsupervised in this way
        real = real.view(-1, 784).to(device) # flatten the image
        batch_size = real.shape[0]
        
        # Train Discriminator: max log(D(real)) + log(1 - D(G(z))), z as the noise
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1) # flatten the output with the view operation
        # See the documentation of BCELoss: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
        # Note here we set the y_n as 1, and thus the second term in the BCELoss is 0, and we take disc_real
        # as x_n, also BCELoss is the negative log likelihood, so we take the negative of the BCELoss,
        # and thus the lossD_real is -log(D(real)), and we are minimizing this loss.
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) # real images are labeled as 1
        disc_fake = disc(fake).view(-1)
        # See the documentation of BCELoss: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
        # Note here we set the y_n as 0, and thus the first term in the BCELoss is 0, and we take disc_fake
        # as x_n, also BCELoss is the negative log likelihood, so we take the negative of the BCELoss,
        # and thus the lossD_fake is -log(1 - D(G(z))), and we are minimizing this loss.
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # fake images are labeled as 0
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True) # so that we can reuse disc_fake, or we can simply fake.detach()
        opt_disc.step()
        
        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        
        # For the tensorboard
        if batch_idx == 0:
            print(
                f'Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {lossD:.4f}, loss G: {lossG:.4f}'
            )
            
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                
                writer_fake.add_image(
                    "MNIST Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "MNIST Real Images", img_grid_real, global_step=step
                )
                step += 1