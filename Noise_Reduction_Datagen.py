#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchaudio as ta
import torch
from torch.utils.data import Dataset, DataLoader

import os

import librosa

import IPython.display as ipd

import numpy as np

import glob

from random import shuffle


# In[2]:


class Signal_Synthesis_DataGen(Dataset):
    def __init__(self, noise_dir, noise_metadata, signal_dir, signal_metadata, num_samples=200, signal_path_save=None, noise_path_save=None, default_sr=16000, sec=6, augment=False):
        
        self.noise_dir = noise_dir
        self.noise_metadata = noise_metadata
        self.signal_dir = signal_dir
        self.signal_metadata = signal_metadata
        self.signal_path_save = signal_path_save
        self.noise_path_save = noise_path_save
        self.default_sr = default_sr
        self.sec = sec
        self.augment = augment
        
        if os.path.exists(self.noise_path_save):
            noise_paths = np.load(self.noise_path_save)
        else:
            noise_paths = []
            for root, dirs, files in os.walk(noise_dir):
                for name in files:
                    if name.endswith(".wav"):
                        noise_paths.append(os.path.join(root, name))
            noise_paths = np.asarray(noise_paths)
#         shuffle(noise_paths)
        self.noise_paths = noise_paths[:num_samples]
        
        if os.path.exists(self.signal_path_save):
            print("Loading from npy file")
            self.signal_paths = np.load(self.signal_path_save)
            
        else:
            print("Loading from the directoriesy")
            self.signal_paths = self.get_sinal_paths(self.signal_dir)
            
        #self.data_list = self.make_data_list(self.signal_paths, self.noise_paths)
        
        
            
    def get_sinal_paths(self, signal_dir):

        signal_paths = []
        for file in tqdm(os.listdir(signal_dir)):
            if file.endswith(".mp3"):
                signal_paths.append(os.path.join(signal_dir, file))
        return signal_paths
    
    
    
    def get_noise_from_sound(self, signal, noise, SNR):
        
        RMS_s = torch.sqrt(torch.mean(signal**2))

        RMS_n = torch.sqrt(RMS_s**2/pow(10., SNR/10))

        RMS_n_current = torch.sqrt(torch.mean(noise**2))
        noise = noise*(RMS_n/RMS_n_current)

        return noise
    
    
    
    def get_mixed_signal(self, signal: torch.Tensor, noise: torch.Tensor, default_sr, sec, SNR):
        
        snip_audio = np.random.randint(0, 2)
        if snip_audio:
            signal = ta.transforms.Vad(sample_rate=default_sr)(signal)
        
        sig_length = int(default_sr * sec)
        
        if len(signal) > sig_length:
            signal = signal[: sig_length]
        elif len(signal) <= sig_length:
            zero_signal = torch.zeros((signal.size()))
            while len(signal) < sig_length:
                signal = torch.cat((signal, zero_signal))
                zero_signal = torch.zeros(signal.size())
            signal = signal[ : sig_length]
        
    
        noise_len = len(noise)
        signal_len = len(signal)

        if len(noise) > len(signal):
            noise = noise[0 : len(signal)]
        elif len(noise) <= len(signal):

            #noise = torch.cat((noise, torch.zeros((len(signal) - len(noise)))))
            for i in range(int(len(signal)/len(noise))+1):
                noise = torch.cat((noise, noise))

            noise = noise[:len(signal)]

        noise = self.get_noise_from_sound(signal, noise, SNR)

        signal_noise = signal+noise
        return signal_noise, signal
    
    
    
    def get_ids(self, signal_paths, noise_paths, idx):

        signal_id = idx//len(noise_paths)
        noise_id = idx - signal_id*len(noise_paths)
#         print(signal_id, noise_id)
        signal_path, noise_path = signal_paths[signal_id], noise_paths[noise_id]
        
        return signal_path, noise_path
        
    def develop_data(self, signal_path, noise_path):
        
        SNR = np.random.randint(0, np.random.randint(0, 50)+1)
#         print(SNR)
        
        noise, nsr = librosa.load(noise_path, sr=self.default_sr)
        signal, ssr = librosa.load(signal_path, sr=self.default_sr)
        
        noise = torch.from_numpy(noise)
        signal = torch.from_numpy(signal)
        
        signal_noise_add, signal = self.get_mixed_signal(signal, noise, self.default_sr, self.sec, SNR)
        
        return signal_noise_add, signal
        
        
        
    def __len__(self):
    
        return len(self.signal_paths)*len(self.noise_paths)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        signal_path, noise_path = self.get_ids(self.signal_paths, self.noise_paths, idx)
        
        signal_noise_add, signal = self.develop_data(signal_path, noise_path)
        
        return signal_noise_add/signal_noise_add.max(), signal/signal.max()
#         return signal_noise_add, signal
        
    
        
                    
                    


# In[3]:


if __name__ == "__main__":
    noise_dir = "./dataset/UrbanSound8K/audio/"
    noise_metadata = "./dataset/UrbanSound8K/metadata/UrbanSound8K.csv"
    signal_dir = "./dataset/cv-corpus-5.1-2020-06-22/en/clips/"
    signal_metadata = "./dataset/cv-corpus-5.1-2020-06-22/en/train.tsv"
    num_samples = 1000
    signal_save_path = "./signal_paths_save.npy"
    noise_save_path = "./noise_paths_save.npy"
    default_sr = 16000
    sec = 6
    augment=False

    
    signal_synthesis_dataset = Signal_Synthesis_DataGen(noise_dir, noise_metadata, signal_dir, signal_metadata, num_samples, signal_save_path, noise_save_path, default_sr, sec, augment)
    signal_mix, signal = signal_synthesis_dataset.__getitem__(4532)

    x = signal_mix.numpy()
    ipd.Audio(x, rate=default_sr)


# In[ ]:




