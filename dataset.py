import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle, randrange
from torch import tensor, float32, save, load
from torch.utils.data import Dataset
from nilearn.image import load_img, smooth_img, clean_img
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_aal, fetch_atlas_destrieux_2009, \
    fetch_atlas_harvard_oxford
from sklearn.model_selection import StratifiedKFold


class Datasetinsomia(Dataset):
    def __init__(self, sourcedir, roi, k_fold=None, target_feature='label', dynamic_length=None):
        super().__init__()
        self.roi = roi
        self.timeseries_dict = []
        self.num_nodes = []
        for i, r in enumerate(self.roi):
            self.filename = 'insomnia'
            self.filename += f'_{r}'
            self.timeseries_dict.append(load(os.path.join(sourcedir, f'{self.filename}.pth')))
            self.num_nodes.append(list(self.timeseries_dict[i].values())[0].shape[1])
        self.num_timepoints = list(self.timeseries_dict[0].values())[0].shape[0]
        self.dynamic_length = dynamic_length


        # self.num_timepoints, self.num_nodes = list(self.timeseries_dict.values())[0].shape
        # print(self.num_timepoints, self.num_nodes)
        self.full_subject_list = list(self.timeseries_dict[0].keys())
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        behavioral_df = pd.read_csv(os.path.join(sourcedir, 'behavioral', 'insomnia_exp.csv')).set_index('Subject')[target_feature]
        self.num_classes = len(behavioral_df.unique())
        self.behavioral_dict = behavioral_df.to_dict()
        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]
        self.train = None

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        if train: shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [
            self.full_subject_list[idx] for idx in test_idx]
        self.train = train

    def __getitem__(self, idx):  # 小批次
        subject = self.subject_list[idx]
        timeseries_all = {}
        for i, r in enumerate(self.roi):
            timeseries = self.timeseries_dict[i][subject]
            if not self.dynamic_length is None:
                sampling_init = randrange(len(timeseries) - self.dynamic_length + 1)
                timeseries = timeseries[sampling_init: sampling_init + self.dynamic_length]
                timeseries_all[f'timeseries_{r}'] = tensor(timeseries, dtype=float32)

        label = self.behavioral_dict[int(subject)]
        if label == 0:
            label = tensor(0)
        elif label == 1:
            label = tensor(1)
        else:
            raise

        # x = {'id': subject, 'label': label}
        # x.update(timeseries_all)
        return {**{'id': subject, 'label': label}, **timeseries_all}


class DatasetABIDE(Dataset):
    def __init__(self, sourcedir, roi, k_fold=None, target_feature='label', dynamic_length=None):
        super().__init__()
        self.roi = roi
        self.timeseries_dict = []
        self.num_nodes = []
        for i, r in enumerate(self.roi):
            self.filename = 'ABIDENYU'
            self.filename += f'_{r}'
            self.timeseries_dict.append(load(os.path.join(sourcedir, f'{self.filename}.pth')))
            self.num_nodes.append(list(self.timeseries_dict[i].values())[0].shape[1])
        self.num_timepoints = list(self.timeseries_dict[0].values())[0].shape[0]
        self.dynamic_length = dynamic_length


        # self.num_timepoints, self.num_nodes = list(self.timeseries_dict.values())[0].shape
        # print(self.num_timepoints, self.num_nodes)
        self.full_subject_list = list(self.timeseries_dict[0].keys())
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        behavioral_df = pd.read_csv(os.path.join(sourcedir, 'behavioral', 'ABIDENYU.csv')).set_index('Subject')[target_feature]
        self.num_classes = len(behavioral_df.unique())
        self.behavioral_dict = behavioral_df.to_dict()
        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]
        self.train = None

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        if train: shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [
            self.full_subject_list[idx] for idx in test_idx]
        self.train = train

    def __getitem__(self, idx):  # 小批次
        subject = self.subject_list[idx]
        timeseries_all = {}
        for i, r in enumerate(self.roi):
            timeseries = self.timeseries_dict[i][subject]
            if not self.dynamic_length is None:
                # sampling_init = randrange(len(timeseries) - self.dynamic_length + 1)
                sampling_init = 0
                timeseries = timeseries[sampling_init: sampling_init + self.dynamic_length]
                timeseries_all[f'timeseries_{r}'] = tensor(timeseries, dtype=float32)

        label = self.behavioral_dict[int(subject)]
        if label == 1:
            label = tensor(0)
        elif label == 2:
            label = tensor(1)
        else:
            raise

        # x = {'id': subject, 'label': label}
        # x.update(timeseries_all)
        return {**{'id': subject, 'label': label}, **timeseries_all}


class DatasetHCPRest(Dataset):
    def __init__(self, sourcedir, roi, k_fold=None, target_feature='label', dynamic_length=None):
        super().__init__()
        self.filename = 'insomnia'
        self.filename += f'_{roi}'
        self.dynamic_length = dynamic_length
        self.timeseries_dict = load(os.path.join(sourcedir, f'{self.filename}.pth'))

        self.num_timepoints, self.num_nodes = list(self.timeseries_dict.values())[0].shape
        # print(self.num_timepoints, self.num_nodes)
        self.full_subject_list = list(self.timeseries_dict.keys())
        if k_fold is None:
            self.subject_list = self.full_subject_list
        else:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0) if k_fold is not None else None
            self.k = None

        behavioral_df = pd.read_csv(os.path.join(sourcedir, 'behavioral', 'insomnia_exp.csv')).set_index('Subject')[target_feature]
        self.num_classes = len(behavioral_df.unique())
        self.behavioral_dict = behavioral_df.to_dict()
        self.full_label_list = [self.behavioral_dict[int(subject)] for subject in self.full_subject_list]
        self.train = None

    def __len__(self):
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    def set_fold(self, fold, train=True):
        assert self.k_fold is not None
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.full_subject_list, self.full_label_list))[fold]
        if train: shuffle(train_idx)
        self.subject_list = [self.full_subject_list[idx] for idx in train_idx] if train else [
            self.full_subject_list[idx] for idx in test_idx]
        self.train = train

    def __getitem__(self, idx):
        subject = self.subject_list[idx]
        timeseries = self.timeseries_dict[subject]
        # timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / np.std(timeseries, axis=0, keepdims=True)
        if not self.dynamic_length is None:
            # if self.train:
            # sampling_init = randrange(len(timeseries) - self.dynamic_length + 1)
            sampling_init = 0
            timeseries = timeseries[sampling_init: sampling_init + self.dynamic_length]

        label = self.behavioral_dict[int(subject)]

        if label == 0:
            label = tensor(0)
        elif label == 1:
            label = tensor(1)
        else:
            raise

        return {'id': subject, 'timeseries': tensor(timeseries, dtype=float32), 'label': label}
