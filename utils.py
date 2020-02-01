import os
import numpy as np
import cv2
import pandas as pd
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

import constants
from dataset import get_dataloaders


class Trainer:
    def __init__(self, model, optimizer, scheduler, recorder, use_gpu=True):
        self.model, self.optimizer, self.scheduler, self.use_gpu, self.recorder = model, optimizer, scheduler, use_gpu, recorder
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
        self.model.to(device=self.device)
        self.losses = [nn.CrossEntropyLoss() for _ in range(len(constants.CATEGORIES))]

    def fit(self, nb_epochs):
        for i in range(nb_epochs):
            print("Epoch {}".format(i))
            for ds in range(constants.NB_DATASETS):
                print("Dataset {}".format(ds))
                train_loader, valid_loader = get_dataloaders(ds)
                self.train_step(train_loader)
                self.valid_step(valid_loader)
                self.recorder.reset_for_next_epoch()
        self.scheduler.step()

    def train_step(self, train_loader):
        self.model.train()
        for inputs, targets in tqdm(train_loader):
            if self.use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs = inputs.to(device=self.device, dtype=torch.float)
            targets = targets.to(device=self.device, dtype=torch.int64)

            # Forward
            outputs = self.model(inputs)
            all_losses = [l(outputs[k], targets[:, k]) for k, l in enumerate(self.losses)]
            loss = all_losses[0] + all_losses[1] + all_losses[2]
            acc = compute_accuracy(outputs, targets)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.recorder.train_step(inputs.shape[0], all_losses, acc)

    def valid_step(self, valid_loader):
        with torch.no_grad():
            self.model.eval()
            for inputs, targets in tqdm(valid_loader):
                if self.use_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()

                    inputs = inputs.to(device=self.device, dtype=torch.float)
                    targets = targets.to(device=self.device, dtype=torch.int64)

                    # Forward
                    outputs = self.model(inputs)
                    all_losses = [l(outputs[k], targets[:, k]) for k, l in enumerate(self.losses)]
                    acc = compute_accuracy(outputs, targets)

                    self.recorder.valid_step(inputs.shape[0], all_losses, acc)

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(os.getcwd(), constants.WEIGHTS_PATH))


class Recorder:
    def __init__(self):
        self.epochs, self.train_loss, self.val_loss, self.train_acc, self.val_acc, self.valid_steps, self.train_steps = [], [], [], [], [], 0.0, 0.0
        self.current_n, self.current_tot_loss, self.current_correct = copy.deepcopy(constants.RECORDER_TEMPLATE2), copy.deepcopy(constants.RECORDER_TEMPLATE1), copy.deepcopy(constants.RECORDER_TEMPLATE1)

    def train_step(self, n, losses, accuracy):
        self.current_n['train'] += n
        self.train_steps += 1
        for k in range(len(constants.CATEGORIES)):
            self.current_tot_loss['train'][k] += losses[k]
            self.current_correct['train'][k] += accuracy[k]

    def valid_step(self, n, losses, accuracy):
        self.current_n['valid'] += n
        self.valid_steps += 1
        for k in range(len(constants.CATEGORIES)):
            self.current_tot_loss['valid'][k] += losses[k]
            self.current_correct['valid'][k] += accuracy[k]

    def reset_for_next_epoch(self):
        if len(self.epochs) == 0:
            self.epochs.append(0)

        else:
            self.epochs.append(self.epochs[-1] + 1)

        train_loss, val_loss, train_acc, val_acc = [], [], [], []

        for k in range(len(constants.CATEGORIES)):
            train_loss.append(self.current_tot_loss['train'][k] / self.train_steps)
            val_loss.append(self.current_tot_loss['valid'][k] / self.valid_steps)
            train_acc.append(self.current_correct['train'][k] / self.current_n['train'])
            val_acc.append(self.current_correct['valid'][k] / self.current_n['valid'])

        print_results(train_loss, val_loss, train_acc, val_acc)

        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)

        self.reset()

    def reset(self):
        self.current_n, self.current_tot_loss, self.current_correct = copy.deepcopy(constants.RECORDER_TEMPLATE2), copy.deepcopy(constants.RECORDER_TEMPLATE1), copy.deepcopy(constants.RECORDER_TEMPLATE1)
        self.train_steps, self.valid_steps = 0.0, 0.0

    def plot(self):
        save_loss_acc_graph(self.epochs, self.train_loss, self.val_loss, self.train_acc, self.val_acc)


def compute_accuracy(outputs, targets):
    all_accuracy = []
    for i in range(len(constants.CATEGORIES)):
        _, pred = torch.max(outputs[i], 1)
        c = (pred == targets[:, i]).float().squeeze()
        all_accuracy.append(torch.sum(c))

    return all_accuracy


def print_results(train_loss, val_loss, train_acc, val_acc):
    for text, values in zip(["Train", "Validation"], [(train_loss, train_acc), (val_loss, val_acc)]):
        print(text)
        for i, (loss, acc) in enumerate(zip(values[0], values[1])):
            print(constants.CATEGORIES[i] + " loss: {:.4f}".format(loss))
            print(constants.CATEGORIES[i] + " acc: {:.4f}".format(acc))
        print("\n")


def save_loss_acc_graph(epochs, train_loss, val_loss, train_acc, val_acc):
    plt.rcParams["figure.figsize"] = [16, 9]
    fig, axs = plt.subplots(2, 2)
    for k, category in enumerate(constants.CATEGORIES):
        for y, text in zip([train_loss, val_loss, train_acc, val_acc], ['Train loss', 'Val loss', 'Train acc', 'Val acc']):
            all_y = [val[k] for val in y]
            axs[k // 2, k % 2].plot(epochs, all_y, label=text)

        axs[k // 2, k % 2].set_title(category)
        axs[k // 2, k % 2].legend()
    plt.savefig(constants.RESULT_GRAPH)


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(sample, size, pad=16):
    height, width = sample.shape
    ymin, ymax, xmin, xmax = bbox(sample[5:-5, 5:-5] > 80)
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < width - 13) else width
    ymax = ymax + 10 if (ymax < height - 10) else height
    img = sample[ymin:ymax, xmin:xmax]
    img[img < 28] = 0
    lx, ly = xmax - xmin, ymax - ymin
    l = max(lx, ly) + pad
    img = np.pad(img, [((l - ly) // 2,), ((l - lx) // 2,)], mode='constant')
    return cv2.resize(img, (size, size)) / 255.0


def convert_to_pickle():
    for ds in range(constants.NB_DATASETS):
        if not os.path.exists(os.path.join(os.getcwd(), constants.PICKLE_FILES[ds])):
            train_data = pd.read_parquet(os.path.join(constants.DATA_DIR, constants.TRAIN_DATA_FILES[ds]))
            images = np.zeros((train_data.shape[0], constants.SIZE, constants.SIZE), dtype=np.float32)
            all_data = 255.0 - np.resize(train_data.iloc[:, 1:].values.astype(np.float32),
                                     (train_data.shape[0], constants.BASE_SHAPE[0], constants.BASE_SHAPE[1]))

            for idx in range(train_data.shape[0]):
                output = all_data[idx]
                images[idx] = crop_resize(output, constants.SIZE)

            with open(constants.PICKLE_FILES[ds], 'wb') as f:
                pickle.dump(images, f)