import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

"""Test de définition du critique + entrainement pour une  (1D - """

torch.normal(mean=torch.zeros(10), std=torch.ones(10))


class Critic(torch.nn.Module):
    def __init__(self, channels=[512, 512]):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(1, channels[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(channels[0], channels[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(channels[1], 1),
        )

    def forward(self, first_law_sample, second_law_sample):
        nb_sample = len(first_law_sample)
        first_law_sample.reshape((nb_sample, 1))
        second_law_sample.reshape((nb_sample, 1))
        return self.linear_relu_stack(first_law_sample), self.linear_relu_stack(
            second_law_sample
        )


def critic_training(sampler1, sampler2, model, optimizer, n_critic, device):
    pbar = tqdm(range(n_critic))
    historique_distance = []
    for iter in pbar:
        first_law_sample = sampler1(64, device)
        second_law_sample = sampler2(64, device)

        # Compute prediction and loss
        pred1, pred2 = model(first_law_sample, second_law_sample)
        distance = loss_w1(pred1, pred2)
        historique_distance.append(distance)
        # Backpropagation
        optimizer.zero_grad()
        distance.backward()
        optimizer.step()

        for weight_and_biases in model.parameters():
            weight_and_biases.data.clamp_(-0.01, 0.01)

        if iter % 100 == 0:
            pbar.set_description("distance %s" % distance)
    return historique_distance


def loss_w1(pred1, pred2):
    result = torch.mean(torch.abs(torch.sub(pred1, pred2)))
    return result


def sampler1(length_batch, device):
    sample = (
        torch.normal(mean=torch.zeros(length_batch), std=torch.ones(length_batch))
        .to(device)
        .reshape(length_batch, 1)
    )
    return sample


def sampler2(length_batch, device):
    sample = (
        torch.normal(mean=-10 * torch.ones(length_batch), std=torch.ones(length_batch))
        .to(device)
        .reshape(length_batch, 1)
    )
    return sample


# model = Critic()

# learning_rate = 1e-3
# batch_size = 64
# epochs = 5

# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, maximize=True)

# n_critic = 1000

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     model.to(device)
#     histo_distance = critic_training(
#         sampler1, sampler2, model, optimizer, n_critic, device
#     )

# histo_distance = [i.cpu().detach().numpy() for i in histo_distance]

# fig = plt.figure()
# ax = fig.add_subplot(2, 1, 1)
# ax.set_xscale("log")
# plt.plot(histo_distance)
