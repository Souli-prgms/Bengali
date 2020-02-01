import torch

import model as mdl
from utils import convert_to_pickle, Trainer, Recorder
import constants


def train():
    model, recorder = mdl.Classifier(), Recorder()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=constants.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, constants.EPOCHS)
    trainer = Trainer(model, optimizer, scheduler, recorder)

    trainer.fit(constants.EPOCHS)
    trainer.save_model()
    recorder.plot()


def main():
    convert_to_pickle()
    train()


if __name__ == "__main__":
    main()