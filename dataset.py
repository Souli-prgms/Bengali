import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pickle

import constants


class BengaliDataset(Dataset):
    def __init__(self, label_file, index):
        self.label_file = label_file
        self.labels = self.retrieve_labels()
        self.images = None
        with open(constants.PICKLE_FILES[index], 'rb') as file:
            self.images = pickle.load(file)
        self.num_samples = self.images.shape[0]
        self.index = index

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = (self.images[idx] - constants.MEAN) / constants.STD
        labels = self.get_label(idx)
        return torch.from_numpy(img).unsqueeze(0), torch.tensor(labels)

    def retrieve_labels(self):
        labels = pd.read_csv(self.label_file)

        for label in constants.CATEGORIES:
            labels[label] = labels[label].astype('uint8')

        return labels

    def get_label(self, index):
        img_name = "Train_{}".format(index + self.num_samples * self.index)
        labels = []
        for l in constants.CATEGORIES:
            label = self.labels.loc[self.labels['image_id'] == img_name][l].iloc[0]
            labels.append(label)
        return labels


def get_datasets(index):
    ds = BengaliDataset(os.path.join(constants.DATA_DIR, constants.TRAIN_CSV), index)
    nb_train = int((1.0 - constants.VALID_RATIO) * len(ds))
    nb_valid = len(ds) - nb_train
    return torch.utils.data.dataset.random_split(ds, [nb_train, nb_valid])


def get_dataloaders(index):
    train_ds, valid_ds = get_datasets(index)
    return DataLoader(dataset=train_ds, batch_size=constants.BS, shuffle=True), DataLoader(dataset=valid_ds, batch_size=constants.BS, shuffle=False)

