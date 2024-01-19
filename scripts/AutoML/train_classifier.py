import os
import pickle
from tqdm import tqdm
import torch
from classifier import classifier, train_classifier
from soft_label_loss import soft_label_loss
from ALGORITHMS import ALGORITHMS
from torch.utils.data import Dataset, DataLoader
import argparse

from ts_benchmark.utils.random_utils import fix_random_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Args for zero-cost AutoML")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--top_k", type=int, default=5)

args = parser.parse_args()
fix_random_seed()

torch.set_num_threads(3)


class metadataset(Dataset):
    def __init__(self, datasets, labels):
        self.datasets = datasets
        self.labels = labels

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        x = self.datasets[index]
        y = self.labels[index]
        return x, y


if __name__ == "__main__":
    with open("data.pkl", "rb") as f:
        loaded_data = pickle.load(f)

    model = classifier(alg_num=len(ALGORITHMS))
    criterion = soft_label_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    datasets, vectors = zip(*loaded_data)
    mydataset = metadataset(datasets, vectors)

    train_loader = DataLoader(
        dataset=mydataset, batch_size=args.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        dataset=mydataset, batch_size=args.batch_size, shuffle=True
    )

    model_dir = 'ckpt/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    his_loss = 100
    tolerance = 0
    train_noisy_loop = tqdm(range(100), ncols=250, desc="pretrain the classifier")

    for epoch in train_noisy_loop:  # 训练NAC多少个epochs？好像只能凭经验，因为没有test set
        train_loss, train_acc, valid_loss, acc = train_classifier(
            train_loader, valid_loader, model, criterion, optimizer
        )

        train_noisy_loop.set_description(f"Epoch {epoch}:")
        train_noisy_loop.set_postfix(
            train_loss=train_loss, train_acc=train_acc, valid_loss=valid_loss, acc=acc
        )
        with open(model_dir + f"/train_log_{args.exp_id}.txt", "a") as f:
            print(
                f"""Noisy Epoch {epoch}: train loss:{train_loss}, valid loss:{valid_loss}, loss {"decreases and model is saved" if valid_loss < his_loss else "doesn't decrease"}""",
                file=f,
            )
        if valid_loss < his_loss:
            train_noisy_loop.set_description(
                f"valid loss decreases [{his_loss}->{valid_loss}]"
            )

            tolerance = 0
            his_loss = valid_loss
            torch.save(model.state_dict(), model_dir + f"/classifier_{args.exp_id}.pth")
        else:
            tolerance += 1
        if tolerance >= 3:
            break

    model.load_state_dict(torch.load(model_dir + f"/classifier_{args.exp_id}.pth"))
