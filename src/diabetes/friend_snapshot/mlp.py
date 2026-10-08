"""mlp.py - the small PyTorch MLP from the design document (Stage 3)."""
import config  # noqa: F401  thread settings before torch
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score


def make_mlp(d_in, hidden=(64, 32), dropout=(0.3, 0.2)):
    layers, d = [], d_in
    for h, p in zip(hidden, dropout):
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
        d = h
    layers.append(nn.Linear(d, 1))   # NO sigmoid: BCEWithLogitsLoss applies it inside the loss
    return nn.Sequential(*layers)


def train_mlp(Xtr, ytr, Xva, pos_weight=1.0, seed=0, hidden=(64, 32), dropout=(0.3, 0.2),
              lr=1e-3, weight_decay=1e-4, batch=1024, max_epochs=100, patience=10):
    """Train on 90% of the training rows, early-stop on the other 10% (validation stays clean).
    Returns (validation probabilities, info dict with model, scaler, epochs)."""
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(config.THREADS_PER_WORKER)
    fit, es = train_test_split(np.arange(len(ytr)), test_size=0.10, stratify=ytr, random_state=seed)
    sc = StandardScaler().fit(Xtr[fit])
    T = lambda a: torch.tensor(sc.transform(a), dtype=torch.float32)
    Xf, Xe, Xv = T(Xtr[fit]), T(Xtr[es]), T(Xva)
    yf, ye = torch.tensor(ytr[fit], dtype=torch.float32), ytr[es]
    model = make_mlp(Xf.shape[1], hidden, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, epochs = -1.0, None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(yf), generator=gen)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:              # BatchNorm cannot train on a single row
                continue
            opt.zero_grad()
            loss_fn(model(Xf[idx]).squeeze(1), yf[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ap = average_precision_score(ye, torch.sigmoid(model(Xe).squeeze(1)).numpy())
        epochs = epoch + 1
        if ap > best + 1e-4:
            best, bad = ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p_val = torch.sigmoid(model(Xv).squeeze(1)).numpy()
    return p_val, {"model": model, "scaler": sc, "epochs": epochs, "inner_ap": round(best, 4)}
