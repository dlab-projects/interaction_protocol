import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import multiprocessing as mp

from interaction_protocol.models import (
    K,
    DeliberationModelSplitPerModel,
    build_all_data,
    convert_round_robin,
    convert_synchronous,
    df_to_verdict_lists,
    fit_map_split_per_model,
    make_verdict2num_mapper,
    model_idxs,
    verdict2num,
)


class DeliberationModel(nn.Module):
    def __init__(self, M, D, K=5):
        super().__init__()
        self.theta_raw = nn.Parameter(torch.zeros(M, K))
        self.phi_raw   = nn.Parameter(torch.zeros(D, K))
        self.alpha     = nn.Parameter(torch.zeros(M))
        self.gamma     = nn.Parameter(torch.tensor(0.0))

    def forward(self, mi, di, same_prev, E_prev, E_within):
        theta = self.theta_raw - self.theta_raw.mean(dim=1, keepdim=True)
        phi   = self.phi_raw   - self.phi_raw.mean(dim=1, keepdim=True)
        logits = theta[mi] + phi[di] + self.alpha[mi].unsqueeze(1)*same_prev + self.gamma*(E_prev+E_within)
        return torch.log_softmax(logits, dim=1)

def fit_map_blended(exp, lr=1e-2, epochs=50, batch_size=4096,
                    sigma_theta=1.0, sigma_phi=1.0, sigma_alpha=0.5, sigma_gamma=0.5,
                    device="cpu", seed=0):
    torch.manual_seed(seed)
    y = torch.as_tensor(exp["y"], dtype=torch.long, device=device)
    mi = torch.as_tensor(exp["model_idx"], dtype=torch.long, device=device)
    di = torch.as_tensor(exp["dilemma_idx"], dtype=torch.long, device=device)
    sp = torch.as_tensor(exp["same_prev_mat"], dtype=torch.float32, device=device)
    ep = torch.as_tensor(exp["E_prev_mat"], dtype=torch.float32, device=device)
    ew = torch.as_tensor(exp["E_within_mat"], dtype=torch.float32, device=device)

    N, K = sp.shape
    M = int(mi.max().item()) + 1
    D = int(di.max().item()) + 1

    model = DeliberationModel(M, D, K=K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(mi, di, sp, ep, ew, y), batch_size=batch_size, shuffle=True)

    nll = nn.NLLLoss(reduction='mean')
    for _ in range(epochs):
        for mi_b, di_b, sp_b, ep_b, ew_b, y_b in dl:
            logp = model(mi_b, di_b, sp_b, ep_b, ew_b)
            loss = nll(logp, y_b)
            theta, phi, alpha, gamma = model.theta_raw, model.phi_raw, model.alpha, model.gamma
            prior = (theta.pow(2).sum()/(2*sigma_theta**2) +
                     phi.pow(2).sum()/(2*sigma_phi**2) +
                     alpha.pow(2).sum()/(2*sigma_alpha**2) +
                     gamma.pow(2)/(2*sigma_gamma**2))
            loss = loss + prior / N
            opt.zero_grad(); loss.backward(); opt.step()

    return model

class DeliberationModelSplit(nn.Module):
    def __init__(self, M, D, K=5):
        super().__init__()
        self.theta_raw = nn.Parameter(torch.zeros(M, K))   # model × label
        self.phi_raw   = nn.Parameter(torch.zeros(D, K))   # dilemma × label
        self.alpha     = nn.Parameter(torch.zeros(M))      # self-stickiness per model
        self.gamma_prev   = nn.Parameter(torch.tensor(0.0))
        self.gamma_within = nn.Parameter(torch.tensor(0.0))

    def forward(self, mi, di, sp, ep, ew):
        # row-center over labels for ID
        theta = self.theta_raw - self.theta_raw.mean(dim=1, keepdim=True)
        phi   = self.phi_raw   - self.phi_raw.mean(dim=1, keepdim=True)
        logits = (
            theta[mi] +
            phi[di] +
            self.alpha[mi].unsqueeze(1) * sp +
            self.gamma_prev * ep +
            self.gamma_within * ew
        )
        return torch.log_softmax(logits, dim=1)

def fit_map_split(exp, lr=1e-2, epochs=60, batch_size=8192,
                  sigma_theta=1.0, sigma_phi=1.0, sigma_alpha=0.5,
                  sigma_gamma_prev=0.5, sigma_gamma_within=0.5,
                  device=None, seed=0, tol=1e-4, verbose=False, early_stop_patience=10):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    y  = torch.as_tensor(exp["y"], dtype=torch.long, device=device)
    mi = torch.as_tensor(exp["model_idx"], dtype=torch.long, device=device)
    di = torch.as_tensor(exp["dilemma_idx"], dtype=torch.long, device=device)
    sp = torch.as_tensor(exp["same_prev_mat"], dtype=torch.float32, device=device)
    ep = torch.as_tensor(exp["E_prev_mat"], dtype=torch.float32, device=device)
    ew = torch.as_tensor(exp["E_within_mat"], dtype=torch.float32, device=device)

    N, K = sp.shape
    M = int(mi.max().item()) + 1
    D = int(di.max().item()) + 1

    model = DeliberationModelSplit(M, D, K=K).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.NLLLoss(reduction='mean')  # <-- don't overwrite this

    dl = DataLoader(TensorDataset(mi, di, sp, ep, ew, y), batch_size=batch_size, shuffle=True)

    best_obj = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for mi_b, di_b, sp_b, ep_b, ew_b, y_b in dl:
            logp = model(mi_b, di_b, sp_b, ep_b, ew_b)
            nll_val = criterion(logp, y_b)

            theta, phi, alpha = model.theta_raw, model.phi_raw, model.alpha
            gp, gw = model.gamma_prev, model.gamma_within
            prior = (
                theta.pow(2).sum()/(2*sigma_theta**2) +
                phi.pow(2).sum()/(2*sigma_phi**2) +
                alpha.pow(2).sum()/(2*sigma_alpha**2) +
                gp.pow(2)/(2*sigma_gamma_prev**2) +
                gw.pow(2)/(2*sigma_gamma_within**2)
            ) / N

            loss = nll_val + prior

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # monitor objective once per epoch
        with torch.no_grad():
            logp_all = model(mi, di, sp, ep, ew)
            nll_all = criterion(logp_all, y)
            theta, phi, alpha = model.theta_raw, model.phi_raw, model.alpha
            gp, gw = model.gamma_prev, model.gamma_within
            prior_all = (
                theta.pow(2).sum()/(2*sigma_theta**2) +
                phi.pow(2).sum()/(2*sigma_phi**2) +
                alpha.pow(2).sum()/(2*sigma_alpha**2) +
                gp.pow(2)/(2*sigma_gamma_prev**2) +
                gw.pow(2)/(2*sigma_gamma_within**2)
            ) / N
            obj = (nll_all + prior_all).item()

        if verbose:
            if epoch % 10 == 0:
                print(f"epoch {epoch+1}: obj={obj:.6f}, nll={nll_all.item():.6f}")

        # simple early stopping on the full objective
        if obj + tol < best_obj:
            best_obj = obj
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                if verbose:
                    print(f"Early stop at epoch {epoch+1}")
                break

    return model

def _subset_by_dilemmas(exp, d_ids):
    """Repeat legacy dictionary rows according to sampled dilemma multiplicity.

    Despite the historical function name, this performs cluster resampling, not
    Boolean subsetting. A dilemma drawn ``k`` times contributes each of its rows
    exactly ``k`` times.
    """
    counts = np.bincount(
        np.asarray(d_ids, dtype=np.int64),
        minlength=int(exp["dilemma_idx"].max()) + 1,
    )
    row_repetitions = counts[exp["dilemma_idx"]]
    return {k: np.repeat(v, row_repetitions, axis=0) for k, v in exp.items()}

def _one_boot(args):
    exp, dilemmas, seed, epochs, batch, lr, device = args
    rng = np.random.default_rng(seed)
    draw = rng.choice(dilemmas, size=len(dilemmas), replace=True)
    boot = _subset_by_dilemmas(exp, draw)
    m = fit_map_split(boot, device=device, epochs=epochs, batch_size=batch, lr=lr, seed=seed, verbose=False)
    return {
        "gamma_prev": float(m.gamma_prev.item()),
        "gamma_within": float(m.gamma_within.item()),
        "alpha": m.alpha.detach().cpu().numpy().copy(),
        "theta": m.theta_raw.detach().cpu().numpy().copy(),
        "phi": m.phi_raw.detach().cpu().numpy().copy()
    }

def bootstrap_split(exp, B=200, epochs=20, batch_size=4096, lr=2e-2, device="cpu", n_jobs=None, base_seed=0):
    dilemmas = np.unique(exp["dilemma_idx"])
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    args = [(exp, dilemmas, base_seed + b, epochs, batch_size, lr, device) for b in range(B)]
    with mp.Pool(processes=n_jobs) as pool:
        outs = pool.map(_one_boot, args)

    # stack results
    gp = np.array([o["gamma_prev"] for o in outs])
    gw = np.array([o["gamma_within"] for o in outs])
    alpha = np.stack([o["alpha"] for o in outs])
    theta = np.stack([o["theta"] for o in outs])
    #phi   = np.stack([o["phi"]   for o in outs])

    # confidence intervals for scalars
    ci_gp = np.percentile(gp, [2.5, 50, 97.5])
    ci_gw = np.percentile(gw, [2.5, 50, 97.5])
    ci_alpha = np.percentile(alpha, [2.5, 50, 97.5], axis=0)

    return {
        "gamma_prev": gp,
        "gamma_within": gw,
        "alpha": alpha,
        "theta": theta,
        #"phi": phi,
        "ci_prev": ci_gp,
        "ci_within": ci_gw,
        "ci_alpha": ci_alpha
    }


def _one_boot_split(args):
    exp, dilemmas, seed, epochs, batch, lr, device = args
    rng = np.random.default_rng(seed)
    draw = rng.choice(dilemmas, size=len(dilemmas), replace=True)
    boot = _subset_by_dilemmas(exp, draw)
    m = fit_map_split_per_model(boot, device=device, epochs=epochs, batch_size=batch, lr=lr, seed=seed, verbose=False)
    return {
        "gamma_prev": m.gamma_prev_m.detach().cpu().numpy().copy(),
        "gamma_within": m.gamma_within_m.detach().cpu().numpy().copy(),
        "alpha": m.alpha.detach().cpu().numpy().copy(),
        "theta": m.theta_raw.detach().cpu().numpy().copy(),
        "phi": m.phi_raw.detach().cpu().numpy().copy()
    }

def bootstrap_model_split(exp, B=200, epochs=20, batch_size=4096, lr=2e-2, device="cpu", n_jobs=None, base_seed=0):
    dilemmas = np.unique(exp["dilemma_idx"])
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    args = [(exp, dilemmas, base_seed + b, epochs, batch_size, lr, device) for b in range(B)]
    with mp.Pool(processes=n_jobs) as pool:
        outs = pool.map(_one_boot_split, args)

    # stack results
    gp = np.array([o["gamma_prev"] for o in outs])
    gw = np.array([o["gamma_within"] for o in outs])
    alpha = np.stack([o["alpha"] for o in outs])
    theta = np.stack([o["theta"] for o in outs])
    #phi   = np.stack([o["phi"]   for o in outs])

    # confidence intervals for scalars
    ci_gp = np.percentile(gp, [2.5, 50, 97.5], axis=0)
    ci_gw = np.percentile(gw, [2.5, 50, 97.5], axis=0)
    ci_alpha = np.percentile(alpha, [2.5, 50, 97.5], axis=0)

    return {
        "gamma_prev": gp,
        "gamma_within": gw,
        "alpha": alpha,
        "theta": theta,
        #"phi": phi,
        "ci_prev": ci_gp,
        "ci_within": ci_gw,
        "ci_alpha": ci_alpha
    }
