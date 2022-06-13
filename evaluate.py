import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from sklearn import metrics
from typing import Optional, List
import torch
import tqdm


def train(model: nn.Module, criterion: nn.Module, train_iter: DataLoader, optimizer: Optimizer, device:torch.device):
    model.train()
    for feature_vector,length,label_true in tqdm.tqdm(train_iter):
        optimizer.zero_grad()
        label_true = torch.squeeze(label_true).to(device)
        label_pred = model(feature_vector, length, device)
        loss = criterion(label_pred, label_true)
        loss.backward()
        optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(device: torch.device, model: nn.Module, data_iter: DataLoader, labels: Optional[List[str]] = None):
    model.eval()
    labels_true, predictions = [], []
    for feature_vector,length,label_true in tqdm.tqdm(data_iter):
        output = model(feature_vector,length,device)
        predictions += output.argmax(dim=1).tolist()
        labels_true += label_true.tolist()

    if labels:
        print(metrics.classification_report(labels_true, predictions, target_names=labels))

    return metrics.accuracy_score(labels_true, predictions)

def pad_batches(batch,pad_index):
    longest_sequence = max([input.size(0) for input,_,_ in batch])
    new_input = torch.stack([
        F.pad(input, (0, longest_sequence - input.size(0)),value=pad_index) for input,_,_ in batch
    ])
    new_lengths = torch.stack([length for _,length,_ in batch])
    new_target = torch.stack([target for _,_, target in batch])

    return new_input, new_lengths, new_target

