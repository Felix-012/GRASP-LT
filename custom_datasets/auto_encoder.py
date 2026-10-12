import h5py
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from torch.utils.data import Dataset


class AutoencoderKLDataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path
        with h5py.File(h5_file_path, 'r') as hf:
            self.keys = sorted(hf.keys(), key=lambda x: int(x.split('_')[1]))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as hf:
            grp = hf[self.keys[idx]]

            mean = torch.tensor(grp['mean'][...])
            logvar = torch.tensor(grp['logvar'][...])

            parameters = torch.cat([mean, logvar], dim=1)
            latent_dist = DiagonalGaussianDistribution(parameters)

        return AutoencoderKLOutput(latent_dist=latent_dist)

    @staticmethod
    def save_autoencoder_outputs(autoencoder_outputs, h5_file_path):
        with h5py.File(h5_file_path, 'w') as hf:
            for idx, output in enumerate(autoencoder_outputs):
                grp = hf.create_group(f'item_{idx}')
                latent_dist = output.latent_dist
                grp.create_dataset('mean', data=latent_dist.mean.cpu().numpy())
                grp.create_dataset('logvar', data=latent_dist.logvar.cpu().numpy())
