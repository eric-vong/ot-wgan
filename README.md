# Wasserstein Generative Adversal Network

[![Build & Test](https://github.com/eric-vong/ot-wgan/actions/workflows/main.yml/badge.svg)](https://github.com/eric-vong/ot-wgan/actions/workflows/main.yml)
[![Code quality](https://github.com/eric-vong/ot-wgan/actions/workflows/quality.yml/badge.svg)](https://github.com/eric-vong/ot-wgan/actions/workflows/quality.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![GitHub](https://img.shields.io/github/license/eric-vong/ot-wgan)

Research project in Optimal Transport course about Wasserstein GAN \
Authors:
* Adrien Majka
* Eric Vong

Does Wasserstein-GAN approximate Wasserstein distances?
The Wasserstein-GAN paper proposes a proxy for the 1-Wasserstein distance that uses neural networks. While that proxy seems to work for the task of training GANs, it is not well understood whether that approach can approximate, numerically, the Wasserstein distance. In this assignment, you will implement the W-GAN approach to solve OT and benchmark it against other approaches (e.g. Sinkhorn divergence) to study its ability to compute a quantity that is truly similar to “true” optimal transport. You should restrict yourself to low-dimensional settings (e.g. 1/2D) or to settings for which the ground truth OT distance is known (i.e. Gaussians or elliptically contoured distributions).

Dataset: https://www.kaggle.com/datasets/jhoward/lsun_bedroom