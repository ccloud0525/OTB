import numpy as np

from set_encoder.setenc_models import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_classifier(train_loader, valid_loader, classifier, criterion, optimizer):
    train_loss = []
    classifier.train()

    for (
        feature,
        label,
    ) in train_loader:  # 对每个batch
        label = torch.Tensor(label).to(DEVICE)
        label = label.float()
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

        for feature, label in valid_loader:
            label = torch.Tensor(label).to(DEVICE)
            label = label.float()
            outputs = classifier(feature)

            loss = criterion(outputs, label)
            valid_loss.append(loss.item())

    return np.mean(train_loss), np.mean(valid_loss)


class classifier(nn.Module):
    def __init__(self, alg_num=31):
        # 后面要参考下Wei Wen文章的GCN实现
        super(classifier, self).__init__()

        # +2用于表示input和output node

        self.nz = 320
        self.fz = 64
        self.set_transformer = SetPool(
            dim_input=self.nz,
            num_outputs=1,
            dim_output=self.fz,
            dim_hidden=self.fz,
            mode="sabPF",
            ln=True,
        )

        self.pred_fc = nn.Sequential(
            nn.Linear(self.fz, self.fz // 2),
            nn.ReLU(),
            nn.Linear(self.fz // 2, alg_num),
            nn.Softmax(dim=-1),
        )
        nn.init.xavier_uniform_(self.pred_fc[0].weight)
        nn.init.xavier_uniform_(self.pred_fc[2].weight)

    def forward(self, feature):
        feature = feature.to(DEVICE)
        feature = self.set_transformer(feature)

        scores = self.pred_fc(feature).squeeze(1)

        return scores
