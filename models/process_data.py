import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class NormalizationTransform(object):

    def __init__(self, norm_constants):
        self.norm_constants = norm_constants
        self.mean = norm_constants['mean']
        self.std = norm_constants['std']

    def __call__(self, sample):
        """
        Transform the sample by normalizing the 'states' using the provided normalization constants.
        :param sample: dictionary containing {'states', 'actions'}
        :return:
        """
        sample['states'] = self.normalize_state(sample['states'])
        return sample

    def inverse(self, sample):
        """
        Transform the sample by de-normalizing the 'states' using the provided normalization constants.
        :param sample: dictionary containing {'states', 'actions'}
        :return:
        """
        sample['states'] = self.denormalize_state(sample['states'])
        return sample

    def normalize_state(self, state):
        """
        Normalize the state using the provided normalization constants.
        :param state: <torch.tensor> of shape (..., num_channels, 32, 32)
        :return: <torch.tensor> of shape (..., num_channels, 32, 32)
        """
        state = state.float()
        state -= self.mean
        state /= self.std
        return state

    def denormalize_state(self, state_norm):
        """
        Denormalize the state using the provided normalization constants.
        :param state_norm: <torch.tensor> of shape (..., num_channels, 32, 32)
        :return: <torch.tensor> of shape (..., num_channels, 32, 32)
        """
        state_norm = state_norm.float()
        state_norm *= self.std
        state = state_norm + self.mean
        return state
