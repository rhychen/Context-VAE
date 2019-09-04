# Context-VAE
Pytorch implementation of VAE with context latent variable. [1] proposed the idea of learning a 'context' latent variable in addition to the individual z latent variables. [2] builds on
top of it with some tweaks in model and training method.
[1] uses sample mean for the instance pooling layer in the 'statistic network'. I tried Wasserstein barycentre instead of sample mean on the kkanji dataset [3] and found no significant effect on training and model performance.

[1] Edwards H, Storkey A. Towards a Neural Statistician. ICLR 2017.
[2] Luke Hewitt et al. The Variational Homoencoder: Learning to learn high capacity generative models from few examples. UAI 2018. 
[3] Tarin Clanuwat et al. Deep Learning for Classical Japanese Literature. arXiv:1812.01718 
