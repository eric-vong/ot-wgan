import time

import torch
from tqdm import tqdm


class Critic(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1, channels=[64, 64, 64, 64]):
        super().__init__()
        self.ngpu = 1
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_dim, channels[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(channels[0], channels[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(channels[1], channels[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(channels[2], channels[3]),
            torch.nn.ReLU(),
            torch.nn.Linear(channels[3], output_dim),
        )

    def forward(self, first_law_sample, second_law_sample):
        nb_sample = len(first_law_sample)
        first_law_sample.reshape((nb_sample, 1))
        second_law_sample.reshape((nb_sample, 1))
        if self.ngpu > 1:
            output1 = torch.nn.parallel.data_parallel(
                self.linear_relu_stack, first_law_sample, range(self.ngpu)
            )
            output2 = torch.nn.parallel.data_parallel(
                self.linear_relu_stack, second_law_sample, range(self.ngpu)
            )
        else:
            output1 = self.linear_relu_stack(first_law_sample)
            output2 = self.linear_relu_stack(second_law_sample)
        return (output1, output2)


class Sampler:
    def __init__(self, mean, std, length_batch, device):
        self.mean = mean
        self.std = std
        self.length_batch = length_batch
        self.device = device

    def __call__(self):
        sample = (
            torch.normal(
                mean=self.mean * torch.ones(self.length_batch),
                std=self.std * torch.ones(self.length_batch),
            )
            .to(self.device)
            .reshape(self.length_batch, 1)
        )
        return sample


def critic_training(sampler1, sampler2, model, optimizer, n_critic, device):
    pbar = tqdm(range(n_critic))
    historique_distance = []
    t1 = time.time()
    for iter in pbar:
        first_law_sample = sampler1()
        second_law_sample = sampler2()

        # Compute prediction and loss
        pred1, pred2 = model(first_law_sample, second_law_sample)
        distance = loss_w1(pred1, pred2)
        historique_distance.append(distance.item())
        # Backpropagation
        optimizer.zero_grad()
        distance.backward()
        optimizer.step()

        for weight_and_biases in model.parameters():
            weight_and_biases.data.clamp_(-1 / 64, 1 / 64)

        if iter % 100 == 0:
            pbar.set_description("distance %s" % (distance))
    t2 = time.time()
    return historique_distance, t2 - t1


def loss_w1(pred1, pred2):
    result = torch.neg(torch.mean(torch.sub(pred1, pred2)))
    return result
