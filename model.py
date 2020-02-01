import torch.nn as nn

import constants
from layers import xresnet34


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor, self.extractor_out = xresnet34()
        self.head1 = nn.Sequential(nn.Linear(self.extractor_out, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, constants.OUTPUTS[0]))
        self.head2 = nn.Linear(self.extractor_out, constants.OUTPUTS[1])
        self.head3 = nn.Linear(self.extractor_out, constants.OUTPUTS[2])

    def forward(self, x):
        x = self.extractor(x)
        return self.head1(x), self.head2(x), self.head3(x)