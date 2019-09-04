# -*- coding: utf-8 -*-
"""
VAE with context latent variable. [1] proposed the idea of learning a 'context'
latent variable in addition to the individual z latent variables. [2] builds on
top of it with some tweaks in model and training method.

[1] Edwards H, Storkey A. Towards a Neural Statistician. ICLR 2017.
[2] Luke Hewitt et al. The Variational Homoencoder: Learning to learn high capacity
    generative models from few examples. UAI 2018.

Assumptions:
    All prior and posterior distributions are assumed to be diagonal Gaussian

Symbols and meanings
  x : An instance of training data
  D : Dataset subsampled from X, contains multiple instances belonging to the
      same class as x.
  X : Full dataset
  z : Per-instance latent variable (as in vanilla VAE)
  c : Context latent variable
  h : Intermediate, per-instance encoding: x -> h, h -> z, c.
  
"""

import sys
sys.path.append(r'C:\AI, Machine learning')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.datasets as dset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from collections import namedtuple
import util

import os
import time
from datetime import datetime
import zipfile

###############################
# Google Colab setup
###############################
from google.colab import widgets
#from google.colab import drive

in_colab    = True
# Google Drive is mounted as 'gdrive' in Google Colab
gdrive_path = '/content/gdrive/My Drive/'
grid        = widgets.Grid(2, 4)

# Mount Google Drive
#from google.colab import drive
#drive.mount(gdrive_path)

# Download from Colab to local drive
#from google.colab import files
#files.download(<filename>)

###############################
# Misc setup
###############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

timestamp = datetime.now()
timestr   = timestamp.strftime("%d") + timestamp.strftime("%m") +\
            timestamp.strftime("%H") + timestamp.strftime("%M")

# Results & logs directory
output_dir = 'ContextVAE_out_h256c64z64lr3e-3_' + timestr
if in_colab:
    output_dir = gdrive_path + output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Saved model directory
chkpt_dir = 'ContextVAE_chkpt_h256c64z64lr3e-3_' + timestr
if in_colab:
    chkpt_dir = gdrive_path + chkpt_dir
if not os.path.exists(chkpt_dir):
    os.makedirs(chkpt_dir)

# Option to carry on training from a previous session
new_session = False
if not new_session:
    saved_model_path = 'ContextVAE_chkpt_h256c64z64lr3e-3_30041345/epoch_80.pth'
    if in_colab:
        saved_model_path = gdrive_path + saved_model_path
    print("WARNING: Continuing from saved model {}".format(saved_model_path))
    saved_states = torch.load(saved_model_path, map_location=device)

LogMoments = namedtuple("LogMoments", ["mean", "logvar"])
Moments    = namedtuple("Moments", ["mean", "std_dev"])
Losses     = namedtuple("Losses", ["total", "recon", "context_KLD", "instance_KLD"])

train_losses = Losses([], [], [], []) if new_session else saved_states['train_losses']
val_losses   = Losses([], [], [], []) if new_session else saved_states['val_losses']

###############################
# Hyperparameters
###############################

# Model parameters
h_len = 256 if new_session else saved_states['hyperparams']['h_len']
c_len = 64  if new_session else saved_states['hyperparams']['c_len']
z_len = 64  if new_session else saved_states['hyperparams']['z_len']

# Dataset parameters
subsample_size = 5 if new_session else saved_states['hyperparams']['subsample_size']
assert (subsample_size > 0), "Subsampled dataset D must have at least 1 sample"

# Initial scaling factor for KL loss
beta = 1

# Training params
batch_size = 128
num_epochs = 400
lr         = 3e-4
starting_epoch = 1 if new_session else saved_states['hyperparams']['epoch']

###############################
# Dataset
###############################

if in_colab:
    archive_path = gdrive_path + 'kkanji2_split.zip'
    data_path    = '.'
    train_data   = 'kkanji2_split/train'
    val_data     = 'kkanji2_split/validation'
    
    zip_ref = zipfile.ZipFile(archive_path, 'r')
    zip_ref.extractall(data_path)
    zip_ref.close()
else:
    train_data = 'C:/datasets/kkanji2_split/train'
    val_data   = 'C:/datasets/kkanji2_split/validation'

# kkanji2 dataset has 3818 characters and 140,384 images.
NUM_TRAIN = 120000
NUM_VAL   =  20384

# To generate subsampled dataset D we need a data structure that facilitates
# sampling in any given class.
class Class2Image(dict):
    def __missing__(self, key):
        self[key] = []
        return self[key]

class CustomImageFolder(dset.ImageFolder):
    """
     Extend ImageFolder dataset to return both a subsampled dataset D and a
     single sample x for training VAE with context latent variable as described
     in Variational Homoencoder (https://arxiv.org/abs/1807.08919). All samples
     in D belong to the same class as x.
     
     As with ImageFolder the images are assumed to be arranged in this way:
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
        
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, transforms.RandomCrop.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list)     : List of the class names (i.e. folder names).
        class_to_idx (dict): Dict with items {class_name: class_index}.
        imgs (list)        : List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=pil_l_loader, subsample_size=None):
        super().__init__(root=root, transform=transform,
                         target_transform=target_transform, loader=loader)
        self.imgs   = self.samples
        self.D_size = subsample_size

        # A dict mapping class labels to image dataset indices to facilitate
        # sampling from a given class.
        self.class2img = Class2Image()       
        for idx, (_, class_idx) in enumerate(self.samples):
            self.class2img[class_idx].append(idx)

    def __getitem__(self, index):
        """        
        Args:
            index (int): Index
        Returns:
            tuple (list): ((Sub-samples), sample, class_index)
        """
        # This is what ImageFolder normally returns
        x, class_idx = super().__getitem__(index)
        # From this single sample create a suitable subsampled dataset (i.e.
        # some other samples from the same class)
        candidates = self.class2img[class_idx]
        class_size = len(candidates)
        if (self.D_size == 1):
            # Sanity check: when a single sample is used to generate 'context'
            # and 'instance' latent codes the model is similar to vanilla VAE and
            # should get similar test result.
            D_indices = [index]
        elif (class_size > self.D_size):
            # Don't let the instance sample x be included in D as well.
            # 'index' is the dataset index. It's simple to find the index for x
            # in the candidates list since the dataset indices contained in
            # candidates are consecutive.
            # Draw random samples. 
            # Pytorch v1.0 doesn't have an equivalent of np.random.choice, but
            # we could do (indices[i] for i in torch.randperm(len(indices)).
            D_indices = np.random.choice(candidates, size=self.D_size, replace=False)
        else:
            # Sample with replacement to get to the required num of samples. A constant sized
            # D for all classes facilitates mini-batch training and simplifies the code.
            D_indices = np.random.choice(candidates, size=self.D_size, replace=True)
        D = []
        for D_idx in D_indices:
            d, D_class_idx = super().__getitem__(D_idx)
            assert (D_class_idx == class_idx),  "D samples must belong to the same class as x."
            D.append(d)
        # Return D as a list of tensors, each size (1 x H x W), rather than
        # concat into one (D_size x H x W) tensor. When used with DataLoader
        # the default collate_fn will add the batch dim so we will get
        # (batch_size x 1 x H x W) tensors and not (batch_size x D_size x H x W).
        return D, x, class_idx, class_size

# Training set
train_dataset = CustomImageFolder(root=train_data,
                                  transform=T.ToTensor(),
                                  loader=pil_l_loader,
                                  subsample_size=subsample_size
                                 )
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

# Validation set
val_dataset = CustomImageFolder(root=val_data,
                                transform=T.ToTensor(),
                                loader=pil_l_loader,
                                subsample_size=subsample_size
                               )
val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

###############################
# Model
###############################

# Shared encoder
#   Input:  x
#   Output: h, tensor of dim (batch_size, h_len)
class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
                           nn.Conv2d(1, 32, kernel_size=6, stride=2),
                           nn.BatchNorm2d(32),
                           nn.LeakyReLU(inplace=True), # Default negative slope is 0.01
                           nn.Conv2d(32, 64, kernel_size=6, stride=2),
                           nn.BatchNorm2d(64),
                           nn.LeakyReLU(inplace=True),
                           nn.Conv2d(64, 128, kernel_size=6, stride=2),
                           nn.BatchNorm2d(128),
                           nn.LeakyReLU(inplace=True),
                           nn.Conv2d(128, 256, kernel_size=4, stride=2),
                           nn.BatchNorm2d(256),
                           nn.LeakyReLU(inplace=True),
                           Flatten(),
                       )

    def forward(self, x):
        h = self.network(x)
        return h
    
# Statistic network q(c|D)
#   Input:  D, a list of tensors with dim (batch_size, 1, H, W)
#   Output: c
class q_c(nn.Module):
    def __init__(self, c_len, h_len):
        super().__init__()
        
        self.h_len = h_len
        self.c_len = c_len

        self.encoder    = SharedEncoder()
        self.fc_prepool = nn.Sequential(
                               nn.Linear(self.h_len, self.h_len),
                               nn.BatchNorm1d(self.h_len),
                               nn.LeakyReLU(inplace=True),
                          )
        self.fc_mean    = nn.Sequential(
                               nn.Linear(self.h_len, self.c_len),
                               nn.BatchNorm1d(self.c_len),
                               nn.LeakyReLU(inplace=True),
                               nn.Linear(self.c_len, self.c_len),
                               nn.BatchNorm1d(self.c_len),
                               nn.LeakyReLU(inplace=True),
                          )
        self.fc_var     = nn.Sequential(
                               nn.Linear(self.h_len, self.c_len),
                               nn.BatchNorm1d(self.c_len),
                               nn.LeakyReLU(inplace=True),
                               nn.Linear(self.c_len, self.c_len),
                               nn.BatchNorm1d(self.c_len),
                               nn.LeakyReLU(inplace=True),
                          )
        
    def forward(self, D, use_barycentre=True):
        if (use_barycentre):
            # Use the Wasserstein barycentre of the D input distributions as
            # context. This computation of the barycentre assumes diagonal
            # Gaussians.
            mean_sum = 0
            sd_sum   = 0
            for d in D:
                h = self.encoder(d)
                v = self.fc_prepool(h)
                mean_sum += self.fc_mean(v)
                logvar    = self.fc_var(v)
                sd_sum   += torch.sqrt(logvar.exp())
            mean    = mean_sum / len(D)
            std_dev = sd_sum / len(D)
            c     = Moments(mean, std_dev)
        else:
            # Per-instance encoding
            e = 0
            for d in D:
                h  = self.encoder(d)
                e += self.fc_prepool(h)
            # Pooling: average over D
            v      = e / len(D)
            # Post-pooling network to get mean and var of diagonal Gaussian
            mean   = self.fc_mean(v)
            logvar = self.fc_var(v)
            c      = LogMoments(mean, logvar)
        return c

# Inference network q(z|x, c)
#   Input:  x, c
#   Output: z
class q_z(nn.Module):
    def __init__(self, c_len, h_len, z_len):
        super().__init__()

        self.c_h_len = c_len + h_len
        self.h_len   = h_len
        self.z_len   = z_len
        
        self.encoder    = SharedEncoder()
        self.fc_network = nn.Sequential(
                               nn.Linear(self.c_h_len, self.h_len),
                               nn.BatchNorm1d(self.h_len),
                               nn.LeakyReLU(inplace=True),
                               nn.Linear(self.h_len, self.h_len),
                               nn.BatchNorm1d(self.h_len),
                               nn.LeakyReLU(inplace=True),
                          )
        self.fc_mean    = nn.Linear(self.h_len, self.z_len)
        self.fc_var     = nn.Linear(self.h_len, self.z_len)
        
    def forward(self, x, c):
        # The (intermediate) encoder is shared with Statistic network,
        # and its output h is concatenated with the context c to feed
        # into the network that produces mean and var for z.
        h      = self.encoder(x)
        c_h    = torch.cat((c, h), dim=1)
        v      = self.fc_network(c_h)
        mean   = self.fc_mean(v)
        logvar = self.fc_var(v)
        z      = LogMoments(mean, logvar)
        return z

# Latent decoder p(z|c)
#   Input:  c
#   Output: z
class p_z(nn.Module):
    def __init__(self, c_len, z_len):
        super().__init__()
        
        self.c_len = c_len
        self.z_len = z_len

        self.network = nn.Sequential(
                               nn.Linear(self.c_len, self.c_len),
                               nn.BatchNorm1d(self.c_len),
                               nn.LeakyReLU(inplace=True),
                               nn.Linear(self.c_len, self.c_len),
                               nn.BatchNorm1d(self.c_len),
                               nn.LeakyReLU(inplace=True),
                       )
        self.fc_mean = nn.Linear(self.c_len, self.z_len)
        self.fc_var  = nn.Linear(self.c_len, self.z_len)

    def forward(self, c):
        v      = self.network(c)
        mean   = self.fc_mean(v)
        logvar = self.fc_var(v)
        z      = LogMoments(mean, logvar)
        return z

# Observation decoder p(x|z, c)
#   Input:  z, c
#   Output: x_recon
class p_x(nn.Module):
    def __init__(self, c_len, z_len):
        super().__init__()
        
        self.c_z_len = c_len + z_len

        self.network = nn.Sequential(
                           nn.Linear(self.c_z_len, 512),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm1d(512),
                           Unflatten(-1, 512, 1, 1),
                           nn.ConvTranspose2d(512, 128, kernel_size=5, stride=2),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm2d(128),
                           nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm2d(64),
                           nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                           nn.ReLU(inplace=True),
                           nn.BatchNorm2d(32),
                           nn.ConvTranspose2d(32, 1, kernel_size=6, stride=2),
                           # For binarised image use sigmoid function to get the probability
                           nn.Sigmoid(),
                       )
        
    def forward(self, z, c):
        c_z     = torch.cat((c, z), dim=1)
        x_recon = self.network(c_z)
        return x_recon
        
class ContextVAE(nn.Module):
    def __init__(self, c_len, z_len, h_len):
        super().__init__()
        
        self.c_len = c_len
        self.z_len = z_len
        self.h_len = h_len
        
        self.q_c = q_c(self.c_len, self.h_len)
        self.q_z = q_z(self.c_len, self.h_len, z_len)
        self.p_z = p_z(self.c_len, self.z_len)
        self.p_x = p_x(self.c_len, self.z_len)
        
    def reparameterize(self, moments):
        if isinstance(moments, LogMoments):
            # Standard deviation
            sd = torch.exp(0.5 * moments.logvar)
        else:
            sd = moments.std_dev
        # We assume the posterior is a multivariate Gaussian
        eps    = torch.randn_like(sd)
        sample = eps.mul(sd).add(moments.mean)
        return sample
        
    def loss_fn(self, x, x_recon, c_post, z_post, z_prior, class_size, beta=1):
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # Context KL divergence
        #   KL( q(c|x) | p(c) )
        # We assume a standard Gaussian prior and diagonal Gaussian posterior,
        # giving the following closed-form KL divergence:
        #   - 0.5 * sum(1 + log(sd ** 2) - mean ** 2 - sd ** 2)
        if isinstance(c_post, LogMoments):
            context_KLD = -0.5 * torch.sum( (1 + c_post.logvar - c_post.mean ** 2 -
                                             c_post.logvar.exp()) / class_size )
        else:
            context_KLD = -0.5 * torch.sum( (1 + torch.log(c_post.std_dev ** 2) -
                                             c_post.mean ** 2 -
                                             c_post.std_dev ** 2) / class_size )
            
        # Per-instance KL divergence
        # Both prior and posterior are assumed to be diagonal Gaussian, giving
        # the following closed-form KL divergence:
        #   KL( q(z|x, c) | p(z|c) )
        #       = - 0.5 * sum(1 + q_logvar - p_logvar - 
        #                     ((q_mean - p_mean)**2 + torch.exp(q_logvar)) / torch.exp(p_logvar))
        instance_KLD = -0.5 * torch.sum( (1 + z_post.logvar - z_prior.logvar -
                                          ((z_post.mean - z_prior.mean) ** 2 +
                                          z_post.logvar.exp()) / z_prior.logvar.exp()) /
                                         class_size )
        
        total_loss = recon_loss + beta * context_KLD + beta * instance_KLD
        losses     = Losses(total_loss, recon_loss, context_KLD, instance_KLD)
        return losses
        
    def forward(self, x, D, class_size, beta=1):
        # Encoding
        c_post = self.q_c(D)
        c      = self.reparameterize(c_post)
        z_post = self.q_z(x, c)
        z      = self.reparameterize(z_post)
        
        # Decoding
        z_prior = self.p_z(c)
        x_recon = self.p_x(z, c)
        losses  = self.loss_fn(x, x_recon, c_post, z_post, z_prior, class_size, beta)

        return x_recon, losses
        
    def gen_sample(self, c_stats):
        c       = self.reparameterize(c_stats)
        z       = torch.randn_like(c_stats.mean).to(device)  # Gaussian samples
        x_recon = self.p_x(z, c)
        return x_recon

###############################
# Main
###############################

model     = ContextVAE(c_len, z_len, h_len).to(device)
#torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
optimiser = optim.Adam(model.parameters(), lr=lr)
# Learning rate annealing
scheduler = optim.lr_scheduler.StepLR(optimiser, step_size = 20, gamma = 0.5)

if not new_session:
    model.load_state_dict(saved_states['model'])
    optimiser.load_state_dict(saved_states['optimiser'])
    
def train(epoch, beta):
    loss_accum      = 0
    loss_avg        = DecayAverage()
    context_KL_avg  = DecayAverage()
    instance_KL_avg = DecayAverage()
    recon_loss_avg  = DecayAverage()

    start_time = time.time()
    model.train()
    for batch, (D, x, _, class_size) in enumerate(train_loader):
        #print("Training batch {}".format(batch))
        with torch.autograd.detect_anomaly():
            x = x.to(device)
            D = [d.to(device) for d in D]
            class_size = class_size.float().unsqueeze(1).to(device)
            recon, losses = model(x, D, class_size, beta)
            losses.total.backward()
            
            loss_accum += losses.total.item()
            loss_avg.update(losses.total.item())
            recon_loss_avg.update(losses.recon.item())
            context_KL_avg.update(losses.context_KLD.item())
            instance_KL_avg.update(losses.instance_KLD.item())

        optimiser.step()
        optimiser.zero_grad()

    epoch_time = time.time() - start_time
    s = 'Time Taken for Epoch {}: {:.2f}s\n'.format(epoch, epoch_time) +\
        'Epoch {} avg. training loss: {:.3f}\n'.format(epoch, loss_accum / NUM_TRAIN) +\
        '         recon loss: {}, context KLD = {}, instance KLD = {}\n'\
        .format(losses.recon, losses.context_KLD, losses.instance_KLD)
    print(s)
    log_fh.write(s)
    s = '{}, {}, {}, {}\n'.format(loss_avg.value, recon_loss_avg.value, context_KL_avg.value, instance_KL_avg.value)
    tplot_fh.write(s)

    if epoch % 10 == 0:
        tplot_fh.flush()
        print("Epoch {} reconstruction:".format(epoch))
        imgs_numpy = recon.detach().to('cpu').numpy()
        fig        = show_images(imgs_numpy[0:25])
        fig.savefig('{}/train_{}.png'.format(output_dir, str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

    return loss_avg.value, recon_loss_avg.value, context_KL_avg.value, instance_KL_avg.value

def validation(epoch, beta):
    loss_accum      = 0
    loss_avg        = RunningAverage()
    context_KL_avg  = RunningAverage()
    instance_KL_avg = RunningAverage()
    recon_loss_avg  = RunningAverage()
    
    model.eval()
    with torch.no_grad():
        for batch, (D, x, _, class_size) in enumerate(val_loader):
            x = x.to(device)
            D = [d.to(device) for d in D]
            class_size = class_size.float().unsqueeze(1).to(device)
            recon, losses = model(x, D, class_size, beta)        
    
            loss_accum += losses.total.item()
            loss_avg.update(losses.total.item())
            recon_loss_avg.update(losses.recon.item())
            context_KL_avg.update(losses.context_KLD.item())
            instance_KL_avg.update(losses.instance_KLD.item())
        
        s = 'Epoch {} validation loss: {:.3f}\n'.format(epoch, loss_accum / NUM_VAL) +\
            '         recon loss: {}, context KLD = {}, instance KLD = {}\n'\
            .format(losses.recon, losses.context_KLD, losses.instance_KLD)
        print(s)
        log_fh.write(s)
        s = '{}, {}, {}, {}\n'.format(loss_avg.value, recon_loss_avg.value, context_KL_avg.value, instance_KL_avg.value)
        vplot_fh.write(s)

        if epoch % 10 == 0:
            log_fh.flush()
            vplot_fh.flush()
            print("Epoch {} reconstruction:".format(epoch))
            imgs_numpy = recon.detach().to('cpu').numpy()
            fig        = show_images(imgs_numpy[0:25])
            fig.savefig('{}/val_{}.png'.format(output_dir, str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

    return loss_avg.value, recon_loss_avg.value, context_KL_avg.value, instance_KL_avg.value

for epoch in range(starting_epoch,  num_epochs + 1):
    if (epoch % 15 == 0):
        beta = beta * 0.9
    
    scheduler.step()
    train_loss, train_recon_loss, train_context_KL, train_instance_KL = train(epoch, beta)
    val_loss,   val_recon_loss,   val_context_KL,   val_instance_KL   = validation(epoch, beta)
    
    train_losses.total.append(train_loss)
    train_losses.recon.append(train_recon_loss)
    train_losses.context_KLD.append(train_context_KL)
    train_losses.instance_KLD.append(train_instance_KL)
    val_losses.total.append(val_loss)
    val_losses.recon.append(val_recon_loss)
    val_losses.context_KLD.append(val_context_KL)
    val_losses.instance_KLD.append(val_instance_KL)
    
    # Plot losses
    if in_colab:
        losses_fig = plot_loss(grid, starting_epoch, epoch, train_losses, val_losses)

    if epoch % 10 == 0:
        # Save checkpoints
        torch.save({
            'model'        : model.state_dict(),
            'optimiser'    : optimiser.state_dict(),
            'train_losses' : train_losses,
            'val_losses'   : val_losses,
            'hyperparams'  : {'h_len'          : h_len,
                              'z_len'          : z_len,
                              'c_len'          : c_len,
                              'subsample_size' : subsample_size,
                              'epoch'          : epoch}
        }, '{}/epoch_{}.pth'.format(chkpt_dir, epoch))

torch.save({
    'model'        : model.state_dict(),
    'optimiser'    : optimiser.state_dict(),
    'train_losses' : train_losses,
    'val_losses'   : val_losses,
    'hyperparams'  : {'h_len'          : h_len,
                      'z_len'          : z_len,
                      'c_len'          : c_len,
                      'subsample_size' : subsample_size,
                      'epoch'          : epoch}
}, '{}/epoch_{}.pth'.format(chkpt_dir, num_epochs))

losses_fig.savefig('{}/losses.png'.format(output_dir), bbox_inches='tight')
plt.close(losses_fig)

log_fh.close()
tplot_fh.close()
vplot_fh.close()