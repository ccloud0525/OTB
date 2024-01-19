import random

import numpy as np
import torch
from tqdm import tqdm
from .set_encoder.setenc_models import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_classifier(
    data_loader, valid_loader, classifier, criterion, optimizer
):
    train_loss = []
    classifier.train()
    train_dataloader = data_loader

    train_dataloader.shuffle()

    for (
        feature,
        label,
    ) in train_dataloader.get_iterator():  # 对每个batch
        label = torch.Tensor(label).to(DEVICE)
        outputs = classifier(feature)
        loss = criterion(outputs, label)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # eval
    with torch.no_grad():
        classifier = classifier.eval()
        valid_loss = []

        for i, (feature, label) in enumerate(valid_loader.get_iterator()):
            label = torch.Tensor(label).to(DEVICE)

            outputs = classifier(feature)

            loss = criterion(outputs, label)
            valid_loss.append(loss.item())

    return np.mean(train_loss), np.mean(valid_loss)


def evaluate(test_loader, classifier, criterion):
    classifier = classifier.eval()
    with torch.no_grad():
        test_loss = []
        for i, (feature, label) in enumerate(test_loader.get_iterator()):
            label = torch.Tensor(label).to(DEVICE)

            outputs = classifier(feature)

            loss = criterion(outputs, label)
            test_loss.append(loss.item())

    return np.mean(test_loss)


class classifier(nn.Module):
    def __init__(self, alg_num=31):
        # 后面要参考下Wei Wen文章的GCN实现
        super(classifier, self).__init__()

        # +2用于表示input和output node

        self.nz = 320
        self.fz = 128
        self.set_transformer = SetPool(
            dim_input=self.nz,
            num_outputs=1,
            dim_output=self.nz,
            dim_hidden=self.nz,
            mode="sabP",
        )

        self.pred_fc = nn.Sequential(
            nn.Linear(self.nz, self.fz),
            nn.ReLU(),
            nn.Linear(self.nz, self.fz // 2),
            nn.ReLU(),
            nn.Linear(self.fz // 2, alg_num),
            nn.Softmax(),
        )

    def forward(self, feature):
        feature = feature.to(DEVICE)
        feature = self.set_transformer(feature)

        scores = self.pred_fc(feature)

        return scores
