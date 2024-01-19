import torch


def soft_label_loss(output, label):
    K = label.shape[-1]

    loss = torch.sum(label + 1 / K * output - label * output, dim=-1)

    loss = torch.mean(loss)

    return loss
