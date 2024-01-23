from abc import ABC, abstractmethod
import math

import torch
import torch.nn.functional as F


class Scheduler(ABC):
    def __init__(self, beta_min, beta_max, N):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N

    def get_discrete_betas(self):
        ts = torch.linspace(0, 1, self.N)
        betas = self.get_sde_beta(ts)
        return betas / self.N

    @abstractmethod
    def get_sde_beta(self, t):
        pass


class Linear(Scheduler):
    def get_sde_beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)


class Sigmoid(Scheduler):
    def __init__(self, beta_min, beta_max, N, temperature=0.1):
        self.temperature = temperature
        super().__init__(beta_min, beta_max, N)

    def get_sde_beta(self, t):
        return F.sigmoid((t - 0.5)/self.temperature) * (self.beta_max - self.beta_min) + self.beta_min


class Cosine(Scheduler):
    def get_sde_beta(self, t):
        return (self.beta_max - self.beta_min) * (1 - torch.cos(math.pi * t)) / 2 + self.beta_min


class Quadratic(Scheduler):
    def get_sde_beta(self, t):
        return -(self.beta_max - self.beta_min) * torch.pow(t, 2) + self.beta_max
    

scheduler_dict = {
    'linear': Linear,
    'sigmoid': Sigmoid,
    'cosine': Cosine,
    'quadratic': Quadratic,
}