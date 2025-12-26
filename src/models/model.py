# src/models/model.py

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class RepairDataset(Dataset):
    def __init__(self, X_cat, X_num, y):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


class RepairEmbNet(nn.Module):
    def __init__(self, cat_cardinalities, emb_dims, n_num, hidden=128, p=0.2):
        super().__init__()
        self.cat_cols = list(cat_cardinalities.keys())

        self.emb = nn.ModuleDict({
            col: nn.Embedding(cat_cardinalities[col], emb_dims[col])
            for col in self.cat_cols
        })

        emb_total = sum(emb_dims.values())
        self.bn_num = nn.BatchNorm1d(n_num) if n_num > 0 else None

        self.fc1 = nn.Linear(emb_total + n_num, hidden)
        self.drop1 = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.drop2 = nn.Dropout(p)
        self.out = nn.Linear(hidden // 2, 1)

    def forward(self, x_cat, x_num):
        embs = [self.emb[col](x_cat[:, i]) for i, col in enumerate(self.cat_cols)]
        x = torch.cat(embs, dim=1)

        if self.bn_num is not None:
            x_num = self.bn_num(x_num)
        x = torch.cat([x, x_num], dim=1)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.out(x)
