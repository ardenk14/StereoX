import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision.io import read_image

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
    

def process_stereo_data(path_to_dataset, batch_size=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'states': x_t,x_{t+1}, ... , x_{t+num_steps}
     'actions': u_t, ..., u_{t+num_steps-1},
    }
    where:
     states: torch.float32 tensor of shape (batch_size, num_steps+1, state_size)
     actions: torch.float32 tensor of shape (batch_size, num_steps, state_size)

    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.

    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.

    :return train_loader: <torch.utils.data.DataLoader> for training
    :return val_loader: <torch.utils.data.DataLoader> for validation
    :return normalization_constants: <dict> containing the mean and std of the states.
    """
    train_data = None
    val_data = None
    normalization_constants = {
        'mean': None,
        'std': None,
    }
    d_set = StereoDataset(path_to_dataset)

    train_data, val_data = random_split(d_set, [0.8, 0.2])

    lst = []
    for i in train_data:
        lst.append(i['L_img'])
        #print("LEFT: ", i["L_img"].shape)
        lst.append(i['R_img'])
        #print("RIGHT: ", i["R_img"].shape)
    t = torch.cat(lst, dim=-1).float()

    t = t.flatten()
    std, mean = torch.std_mean(t)

    normalization_constants['mean'] = mean
    normalization_constants['std'] = std

    norm_tr = NormalizationTransform(normalization_constants)
    train_data.dataset.transform = norm_tr
    val_data.dataset.transform = norm_tr

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader, normalization_constants


class StereoDataset(Dataset):

    def __init__(self, path_to_dataset):
        self.path = path_to_dataset
        self.length = 200
        self.rs = torchvision.transforms.Resize((37, 122))#(185, 610)

    def __len__(self):
        return 200
    
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem_(i)

    def __getitem__(self, item):
        sample = {
            'L_img': None,
            'R_img': None,
        }
        s = str(item).zfill(6) + '_1' + str(item%2) + '.png'
        path_L = self.path + "image_L/" + s
        path_R = self.path + "image_R/" + s
        img_L = self.rs(read_image(path_L))
        img_R = self.rs(read_image(path_R))
        sample['L_img'] = img_L[:, :370, :1220].float() #185, 610
        sample['R_img'] = img_R[:, :370, :1220].float()
        #traj_idx = item // self.trajectory_length
        #step_idx = item % self.trajectory_length

        #state = self.data[traj_idx]['states'][step_idx:step_idx + self.num_steps + 1].astype(np.float32)
        #action = self.data[traj_idx]['actions'][step_idx:step_idx + self.num_steps].astype(np.float32)

        #sample['states'] = torch.permute(torch.from_numpy(state), (0, 3, 1, 2))
        #sample['actions'] = torch.from_numpy(action)

        #if self.transform is not None:
        #    sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    d = StereoDataset('asdf')
    for i in range(10):
        d.__getitem__(i)