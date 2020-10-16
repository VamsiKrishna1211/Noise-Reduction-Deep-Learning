#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchaudio as ta
import torch
from torch.utils.data import Dataset, DataLoader

import os

import librosa
import pandas as pd
import IPython.display as ipd

import numpy as np

import glob
import gc
from random import shuffle
from tqdm.auto import tqdm


# In[2]:


class Signal_Synthesis_DataGen(Dataset):
    def __init__(self, noise_dir, signal_dir, signal_nums_save=None, num_noise_samples=None, num_signal_samples=None, noise_path_save=None,\
                 n_fft=400, win_length=400, hop_len=200, f_min=0, f_max=8000, \
                 perform_stft=True, normalize=True, default_sr=16000, sec=6, augment=False):

        self.noise_dir = noise_dir
        self.signal_dir = signal_dir
        self.signal_nums_save = signal_nums_save
        self.num_noise_samples = num_noise_samples
        self.num_signal_samples = num_signal_samples
        self.noise_path_save = noise_path_save
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_len = hop_len
        self.f_min = f_min
        self.f_max = f_max
        self.perform_stft = perform_stft
        self.normalize = normalize
        self.default_sr = default_sr
        self.sec = sec
        self.augment = augment



        if os.path.exists(self.noise_path_save):
            print("Loading noise from saved file")
            noise_paths = np.load(self.noise_path_save)
        else:
            noise_paths = []
            for root, dirs, files in os.walk(noise_dir):
                for name in files:
                    if name.endswith(".wav"):
                        noise_paths.append(os.path.join(root, name))
            noise_paths = np.asarray(noise_paths)
#         shuffle(noise_paths)
        # print(self.num_noise_samples)
        if self.num_noise_samples is not None:
            self.noise_paths = noise_paths[:self.num_noise_samples]
        else:
            self.noise_paths = noise_paths
        if os.path.exists(signal_nums_save):
            print("Loading nums from npy file")
            self.signal_nums = torch.from_numpy(np.load(signal_nums_save))
        else:
            self.signal_nums = torch.from_numpy(self.get_signal_paths(signal_dir))

        if self.num_signal_samples is not None:
            self.signal_nums = self.signal_nums[:self.num_signal_samples]
        print(len(self.signal_nums))
        self.prefix = "common_voice_en_"
        self.suffix = ".mp3"




    def get_signal_paths(self, clips_path):

        file_nums = []
        for file in tqdm(os.listdir(clips_path)):
            num = file.split("_")[3]
            num = int(num.split(".")[0])
            file_nums.append(num)
        file_nums = np.asarray(file_nums)
        return file_nums



    def get_noise_from_sound(self, signal, noise, SNR):

        RMS_s = np.sqrt(np.mean(signal**2))

        RMS_n = np.sqrt(RMS_s**2/pow(10., SNR/10))

        RMS_n_current = np.sqrt(np.mean(noise**2))
        noise = noise*(RMS_n/RMS_n_current)

        return noise



    def get_mixed_signal(self, signal: torch.Tensor, noise: torch.Tensor, default_sr, sec, SNR):

        snip_audio = np.random.randint(0, 2)
        # if snip_audio:
        #     signal = ta.transforms.Vad(sample_rate=default_sr)(signal)

        sig_length = int(default_sr * sec)

        if len(signal) > sig_length:
            signal = signal[: sig_length]
        elif len(signal) <= sig_length:
            zero_signal = np.zeros((signal.shape))
            while len(signal) < sig_length:
                signal = np.concatenate((signal, zero_signal))
                zero_signal = np.zeros(signal.shape)
            signal = signal[ : sig_length]


        noise_len = len(noise)
        signal_len = len(signal)

        if len(noise) > len(signal):
            noise = noise[0 : len(signal)]
        elif len(noise) <= len(signal):

            #noise = torch.cat((noise, torch.zeros((len(signal) - len(noise)))))
            for i in range(int(len(signal)/len(noise))+1):
                noise = np.concatenate((noise, noise))

            noise = noise[:len(signal)]

        noise = self.get_noise_from_sound(signal, noise, SNR)

        signal_noise = signal+noise
        return signal_noise, signal

    def construct_signal_path(self, signal_id):

        file_num = self.signal_nums[signal_id]
        if torch.is_tensor(file_num):
            # print("Enter_tensor")
            file_num = file_num.item()
        file_name = self.prefix + str(file_num) + self.suffix
        path = os.path.join(self.signal_dir, file_name)
        if os.path.exists(path):
            return path
        else:
            raise FileExistsError(f"{path}")



    def get_ids(self, signal_paths, noise_paths, idx):

        signal_id = idx//len(noise_paths)
        noise_id = idx - signal_id*len(noise_paths)
#         print(signal_id, noise_id)

        signal_path, noise_path = self.construct_signal_path(signal_id), noise_paths[noise_id]

        signal_noise_add, signal = self.develop_data(signal_path, noise_path)

        return signal_noise_add, signal

    def develop_data(self, signal_path, noise_path):

        SNR = np.random.randint(0, np.random.randint(0, 50)+1)
#         print(SNR)

        noise, nsr = librosa.load(noise_path, sr=self.default_sr)
        signal, ssr = librosa.load(signal_path, sr=self.default_sr)
        # noise, nsr = ta.load(noise_path)
        # noise = ta.transforms.Resample(orig_freq=nsr, new_freq=self.default_sr)(noise)
        # signal, ssr = ta.load(signal_path)
        # signal = ta.transforms.Resample(orig_freq=ssr, new_freq=self.default_sr)(signal)
        # noise = torch.from_numpy(noise)
        # signal = torch.from_numpy(signal)

        signal_noise_add, signal = self.get_mixed_signal(signal, noise, self.default_sr, self.sec, SNR)
        if self.perform_stft:
            signal_noise_add = librosa.stft(signal_noise_add, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_length)
            signal = librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_length)
            # (signal_noise_add, signal) = torch.stft(combined_signal, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_length, normalized=self.normalize)

        return signal_noise_add, signal



    def __len__(self):

        return len(self.signal_nums)*len(self.noise_paths)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        signal_noise_add, signal = self.get_ids(self.signal_nums, self.noise_paths, idx)
        gc.collect()

#         signal_noise_add, signal = signal_noise_add/signal_noise_add.max(), signal/signal.max()
        # print("returning the values from getitem dataset")
        return signal_noise_add, signal
#         return signal_noise_add, signal







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
