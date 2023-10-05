import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def plot_photons_histograms(
        dataset:np.ndarray, axs, figs, log_ax:bool=False, label='original', 
        title:str=None, data_color:str='crimson', a:float=0.8) -> None:
    
    assert dataset.shape[1]==6

    df = pd.DataFrame(dataset, columns = ['E','X', 'Y', 'dX', 'dY', 'dZ'])
    df.hist(column='E', bins=300, color=data_color, alpha=a, density=True, ax=axs[0,0])
    df.hist(column='X', bins=300, color=data_color, alpha=a, density=True, ax=axs[0,1])
    df.hist(column='Y', bins=300, color=data_color, alpha=a, label=label, density=True, ax=axs[0,2])
    df.hist(column='dX', bins=300, color=data_color, alpha=a, density=True, ax=axs[1,0])
    df.hist(column='dY', bins=300, color=data_color, alpha=a, density=True, ax=axs[1,1])
    df.hist(column='dZ', bins=300, color=data_color, alpha=a, density=True, ax=axs[1,2])
    axs[0,2].legend(loc='best')

    if log_ax:
        axs[0,0].set_yscale('log')
        axs[0,1].set_yscale('log')
        axs[0,2].set_yscale('log')
        axs[1,0].set_yscale('log')
        axs[1,1].set_yscale('log')
        axs[1,2].set_yscale('log')

    if title:
        figs.suptitle(title, fontsize=16)



class PhotonsDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = self.setup_data(data_path)
        self.n_samples = self.data.shape[0]
        self.transform = transform
        self.columns = ['E', 'X', 'Y', 'dX', 'dY', 'dZ']

    def __getitem__(self, index):
        sample = self.data[index]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

    def setup_data(self, data_path):
        photons = np.load(data_path)
        mmsc = MinMaxScaler()
        photons_mm = mmsc.fit_transform(photons)
        return photons_mm