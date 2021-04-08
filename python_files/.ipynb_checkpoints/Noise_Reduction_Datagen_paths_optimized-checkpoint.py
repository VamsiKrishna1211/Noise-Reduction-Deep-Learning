#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchaudio as ta
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os

import IPython.display as ipd

# import librosa

# import librosa

# ta.set_audio_backend("sox_io")


# In[2]:


class Signal_Synthesis_DataGen(Dataset):
    def __init__(self, noise_paths, signal_paths, signal_dir, train=False,\
                 n_fft=400, win_length=400, hop_len=200, create_specgram=False,  \
                 perform_stft=True, normalize=True, default_sr=16000, sec=6, epsilon=1e-5, augment=False, device="cpu"):

        self.noise_paths = noise_paths
        self.signal_paths = signal_paths
        self.signal_dir = signal_dir
        self.train = train
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_len = hop_len
        self.create_specgram = create_specgram
        self.perform_stft = perform_stft
        self.normalize = normalize
        self.default_sr = default_sr
        self.sec = sec
        self.epsilon = epsilon
        self.augment = augment


        self.device = device# if torch.cuda.is_available() else "cpu"

        if self.create_specgram == True and self.perform_stft == True:
            raise Exception("Use only one option out of 'create_specgram' and 'perform_stft'")

        print(len(self.signal_paths), self.signal_paths[1])


    def normalize_signal(self, tensor):
        tensor_min_minus = tensor - tensor.min()
        return tensor_min_minus/(tensor_min_minus.abs().max() + self.epsilon) +self.epsilon


    def get_noise_from_sound(self, signal, noise, SNR):

        RMS_s = torch.sqrt(torch.mean(signal**2))

        RMS_n = torch.sqrt(RMS_s**2/torch.pow(10., SNR/10))

        RMS_n_current = torch.sqrt(torch.mean(torch.square(noise)))
        noise = noise*(RMS_n/RMS_n_current)

        return noise



    def get_mixed_signal(self, signal: torch.Tensor, noise: torch.Tensor, default_sr, sec, SNR):


        signal = signal[self.default_sr*torch.randint(0, 4, size=[1]).squeeze().numpy():]
        sig_length = int(default_sr * sec)
        final_len = int(self.default_sr*self.sec)
        # print(len(signal), sig_length)
        if len(signal) > sig_length:
            # print("enter len if")
            signal = signal[ : sig_length]
        elif len(signal) < sig_length:
            
            add_len = final_len - len(signal)
            zeros_signal = np.zeros(add_len, dtype=np.float32)
            signal = signal.numpy()
            signal = np.append(signal, (zeros_signal))
            signal = torch.from_numpy(signal)


        noise_len = len(noise)
        signal_len = len(signal)

        if len(noise) > len(signal):
            noise = noise[0 : len(signal)]

        elif len(noise) <= len(signal):
            noise_buffer = torch.zeros(final_len, dtype=torch.float32)
            noise = noise#.numpy()
            for i in range(signal_len//noise_len):
                noise_buffer[i*noise_len : (i+1)*noise_len] = noise
            noise_buffer[(i+1)*noise_len:] = noise[:(signal_len - (i+1)*noise_len)]
            noise = noise_buffer
        noise = self.get_noise_from_sound(signal, noise, SNR)

        signal_noise = signal+noise
        return signal_noise, signal

    def construct_signal_path(self, signal_id):

        file_name = self.signal_paths[signal_id]
        if torch.is_tensor(file_name):
            # print("Enter_tensor")
            file_name = file_name.item()
        path = file_name
        if os.path.exists(path):
            return path
        else:
            raise FileExistsError(f"{path}")



    def get_ids(self, idx):

        signal_id = idx//len(self.noise_paths)
        noise_id = idx - signal_id*len(self.noise_paths)

        signal_path, noise_path = self.construct_signal_path(signal_id), self.noise_paths[noise_id]

        signal_noise_add, signal = self.develop_data(signal_path, noise_path)

        if self.normalize:
            signal_noise_add = self.normalize_signal(signal_noise_add)
            signal = self.normalize_signal(signal)

        return signal_noise_add, signal

    def develop_data(self, signal_path, noise_path):

        if not os.path.exists(signal_path):
            raise FileExistsError({signal_path})
        if not os.path.exists(noise_path):
            raise FileExistsError({noise_path})
        SNR = torch.randint(0, high=torch.randint(2, 51, size=[1]).squeeze(), size=[1]).squeeze()

        noise, nsr = ta.load(noise_path)
        noise = noise[0]#.to(self.device)
        noise = noise#.type(torch.float32)
        if nsr != self.default_sr:
            noise = ta.transforms.Resample(orig_freq=nsr, new_freq=self.default_sr)(noise)
        signal, ssr = ta.load(signal_path)
        signal = signal[0]#.to(self.device)
        signal = signal#.type(torch.float32)
        if ssr != self.default_sr:
            signal = ta.transforms.Resample(orig_freq=ssr, new_freq=self.default_sr)(signal)


        signal_noise_add, signal = self.get_mixed_signal(signal, noise, self.default_sr, self.sec, SNR)
        if self.perform_stft:
            signal_noise_add = torch.stft(signal_noise_add, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_length)
            signal = torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_length)[:,:,:]
        elif self.create_specgram:
            spec_transformer = ta.transforms.Spectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_len, normalized=self.normalize)
            signal_noise_add = spec_transformer(signal_noise_add)
            signal = spec_transformer(signal)


        return signal_noise_add.unsqueeze(dim=0), signal.unsqueeze(dim=0)#, noise]



    def __len__(self):

        return len(self.signal_paths)*len(self.noise_paths)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        signal_noise_add, signal = self.get_ids(idx)

        return signal_noise_add, signal







# In[3]:


if __name__ == "__main__":
    noise_dir = "./dataset/UrbanSound8K/audio/"
    noise_metadata = "./dataset/UrbanSound8K/metadata/UrbanSound8K.csv"
    signal_dir = "./dataset/cv-corpus-5.1-2020-06-22/en/clips/"
    signal_metadata = "./dataset/cv-corpus-5.1-2020-06-22/en/train.tsv"
    num_samples = 1000
    use_df = True
    df_path = "./dataset/cv-corpus-5.1-2020-06-22/en/train.tsv"
    signal_save_path = "./signal_paths_save.npy"
    noise_save_path = "./noise_paths_save.npy"
    default_sr = 16000
    sec = 6
    augment=False


    signal_synthesis_dataset = Signal_Synthesis_DataGen(noise_dir, noise_metadata, signal_dir, signal_metadata, num_samples, use_df, df_path, signal_save_path, noise_save_path, default_sr, sec, augment)
    signal_mix, signal = signal_synthesis_dataset.__getitem__(4532)
    print(signal_mix.shape)

    # x = signal_mix.numpy()
    # ipd.Audio(x, rate=default_sr)


# In[ ]:
