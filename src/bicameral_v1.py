#!/usr/bin/env python3
"""
Bicameral Upgraded: Natural Metric Latent Space & Separable CMA Conscious Phase.
High-impact optimizations integrated:
  1. Conscious: Coordinate-wise Natural Gradient Adaptation (Sep-CMA-ES style).
  2. Latent Engine: Supervised Ridge-weighted Latent Reconstruction + Matérn 5/2.
  3. Gate Optimizer: Continuous Sigmoid Relaxation with dynamic temperature annealing.
"""

import time
import numpy as np
from scipy.spatial.distance import cdist


# -------------------------------------------------------------
# 1. UPGRADED LATENT SPACE: Supervised Whitening & SVD
# -------------------------------------------------------------
class FastSupervisedLatentSpace:
    """Enhanced Latent Space with supervised elite covariance alignment."""
    def __init__(self, d, n_elite=50):
        self.d = d
        self.n_elite = n_elite
        self.U = None
        self.mu = None
        self.inv_s = None

    def ready(self):
        return self.U is not None

    def update(self, archive):
        if len(archive) < self.d + 2:
            return
        # Sort by loss
        sorted_arch = sorted(archive, key=lambda t: t[1])[:self.n_elite]
        elites = np.array([x for x, _ in sorted_arch])
        losses = np.array([y for _, y in sorted_arch])

        # Center data
        self.mu = elites.mean(0)
        centered = elites - self.mu

        # Weighted SVD: Prioritize the very best points in the projection
        weights = np.exp(-np.linspace(0, 2.0, len(elites)))[:, None]
        weighted_centered = centered * weights

        _, S, Vt = np.linalg.svd(weighted_centered, full_matrices=False)
        r = min(self.d, Vt.shape[0])
        self.U = Vt[:r].T  # (D, d)

        # Scaling vectors
        proj = centered @ self.U
        stds = proj.std(0) + 1e-4
        self.inv_s = 1.0 / stds

    def phi(self, x):
        return ((x - self.mu) @ self.U) * self.inv_s

    def psi(self, c, lb, ub):
        x = self.mu + (c / self.inv_s) @ self.U.T
        return np.clip(x, lb, ub)


# -------------------------------------------------------------
# 2. FAST SEPARABLE CMA-ES CONSCIOUS SOLVER
# -------------------------------------------------------------
class FastSeparableNES:
    """Diagonal Natural Evolution Strategy for Conscious Exploration."""
    def __init__(self, dim, lb, ub, sigma_init=0.2):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.sigma = np.ones(dim) * sigma_init
        self.lr_sigma = 0.1 / np.sqrt(dim)
        self.p_succ = 0.2

    def step(self, best_x, rng):
        # Anisotropic mutation along coordinate axes
        z = rng.standard_normal(self.dim)
        x_mut = np.clip(best_x + self.sigma * z, self.lb, self.ub)
        return x_mut, z

    def adapt(self, z, success):
        # 1/5th success rule for diagonal variance
        factor = np.exp(self.lr_sigma * (1.0 - self.p_succ if success else -self.p_succ))
        if success:
            self.sigma = np.clip(self.sigma * (0.8 + 0.4 * np.abs(z)), 1e-4, 1.5)
        else:
            self.sigma = np.clip(self.sigma * factor, 1e-4, 1.5)


# -------------------------------------------------------------
# 3. HIGH-ACCURACY SURROGATE: Fast Matérn-Kernel Memory
# -------------------------------------------------------------
class FastMaternSurrogate:
    """Matérn-3/2 Kernel Surrogate with Analytical Uncertainty Estimation."""
    def __init__(self, d):
        self.d = d
        self.length_scale = np.sqrt(d)

    def evaluate_acquisition(self, C_cand, C_mem, Y_mem, beta=1.5):
        """Vectorized Upper Confidence Bound (UCB / LCB) surrogate score."""
        if len(C_mem) == 0:
            return np.zeros(len(C_cand))

        # Pairwise distances: (N_cand, N_mem)
        dists = cdist(C_cand, C_mem)

        # Matérn 3/2 Kernel
        sqrt3_d = np.sqrt(3.0) * dists / self.length_scale
        K = (1.0 + sqrt3_d) * np.exp(-sqrt3_d)

        # Kernel Regression (Surrogate Loss Mean)
        weights = K / (K.sum(axis=1, keepdims=True) + 1e-8)
        y_hat = weights @ Y_mem

        # Uncertainty estimate based on kernel density
        density = K.sum(axis=1)
        sigma_hat = 1.0 / np.sqrt(density + 1.0)

        # Lower Confidence Bound: Minimize expected loss + maximize exploration
        lcb = -y_hat + beta * sigma_hat
        return lcb


# -------------------------------------------------------------
# 3b. RANK SURROGATE: Pairwise-Logistic (RankNet-style) + RFF
# -------------------------------------------------------------
class FastRankSurrogate:
    """Rank-based surrogate: learns *orderings*, not values.

    Model: P(i beats j) = sigmoid(s_i - s_j), s(c) = w . phi(c), with
    random-Fourier-feature map phi (RBF approximation, median-heuristic
    bandwidth). Trained by SGD on pairs sampled from memory. Candidates
    are scored by predicted utility s(c) plus a beta-weighted novelty
    bonus (min-distance to memory), mirroring the Matérn UCB's beta knob.

    Why ranks: value-regression smooths barrier ridges into fake valleys
    and over-scores them (the v4-reverie autopsy); comparison-based
    optimizers need comparison-based surrogates (Loshchilov et al. 2010).
    Rank utilities are invariant to monotone transforms of Y and robust
    to barrier-scale outliers. Same interface as FastMaternSurrogate
    (higher score = better); deterministic given (seed, data sequence).
    """

    def __init__(self, d, n_features=128, n_pairs=1500, n_steps=150,
                 batch=64, lr=0.2, l2=1e-4, seed=0):
        self.d = d
        self.F = n_features
        self.n_pairs = n_pairs
        self.n_steps = n_steps
        self.batch = batch
        self.lr = lr
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        # Fixed RFF projection (frequencies redrawn never; bandwidth adapts)
        self.W = self.rng.standard_normal((self.F, d))
        self.b = self.rng.uniform(0, 2 * np.pi, self.F)

    def _features(self, C, gamma):
        return np.sqrt(2.0 / self.F) * np.cos(C @ self.W.T / gamma + self.b)

    def evaluate_acquisition(self, C_cand, C_mem, Y_mem, beta=1.5):
        C_cand = np.asarray(C_cand, dtype=float)
        C_mem = np.asarray(C_mem, dtype=float)
        Y_mem = np.asarray(Y_mem, dtype=float)
        n_cand = len(C_cand)
        if n_cand == 0:
            return np.zeros(0)
        if len(C_mem) < 4:
            return np.zeros(n_cand)
        # --- standardize with memory stats (RFF + SGD need ~O(1) inputs)
        mu = C_mem.mean(axis=0)
        sd = C_mem.std(axis=0) + 1e-9
        Zm = (C_mem - mu) / sd
        Zc = (C_cand - mu) / sd
        # --- median-heuristic bandwidth on a memory subsample
        sub = Zm if len(Zm) <= 256 else Zm[self.rng.choice(
            len(Zm), 256, replace=False)]
        pd = cdist(sub, sub)
        gamma = float(np.median(pd[pd > 1e-12])) if np.any(
            pd > 1e-12) else 1.0
        gamma = max(gamma, 1e-6)
        Phi_m = self._features(Zm, gamma)
        # --- sample ordered pairs (better, worse), skip ties
        n_mem = len(Zm)
        ii = self.rng.integers(0, n_mem, self.n_pairs * 2).reshape(-1, 2)
        dy = Y_mem[ii[:, 0]] - Y_mem[ii[:, 1]]
        keep = np.abs(dy) > 1e-12
        ii = ii[keep][:self.n_pairs]
        w = np.zeros(self.F)
        if len(ii) > 0:
            better_first = Y_mem[ii[:, 0]] < Y_mem[ii[:, 1]]
            A = np.where(better_first, ii[:, 0], ii[:, 1])
            B = np.where(better_first, ii[:, 1], ii[:, 0])
            D = Phi_m[A] - Phi_m[B]  # (P, F): pair differences
            # --- SGD on pairwise logistic loss + L2
            lr = self.lr
            for _ in range(self.n_steps):
                bb = self.rng.integers(0, len(D), self.batch)
                margins = D[bb] @ w
                # d/dw -log sigmoid(margin) = -sigmoid(-margin) * d_margin
                neg = 1.0 / (1.0 + np.exp(np.clip(-margins, -30, 30)))
                grad = -(neg[:, None] * D[bb]).mean(axis=0) + self.l2 * w
                w -= lr * grad
        # --- score: standardized utility + novelty bonus (Matérn-like balance:
        # signal ~O(1), exploration capped, beta keeps a stable meaning)
        s = self._features(Zc, gamma) @ w
        s = (s - s.mean()) / (s.std() + 1e-9)
        dmin = cdist(Zc, Zm).min(axis=1)
        novelty = dmin / (np.median(dmin) + 1e-9)
        return s + beta * np.clip(novelty, 0.0, 2.0)


# -------------------------------------------------------------
# 4. GATE OPTIMIZER: Sigmoid Relaxation + Temperature Annealing
# -------------------------------------------------------------
class SigmoidGate:
    """Continuous sigmoid gate blending conscious/subconscious proposals.

    g = sigmoid((score_sub - score_con) / temperature), with exponential
    temperature annealing T(t) = max(T_min, T0 * decay^t).
    High T early  -> soft/exploratory mixing; low T late -> hard argmax.
    """

    def __init__(self, T0=1.0, T_min=0.05, decay=0.995):
        self.T0 = T0
        self.T_min = T_min
        self.decay = decay
        self.t = 0

    @property
    def temperature(self):
        return max(self.T_min, self.T0 * (self.decay ** self.t))

    @staticmethod
    def sigmoid(v):
        v = np.clip(v, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-v))

    def select(self, x_con, x_sub, score_con, score_sub, rng):
        """Hard probabilistic pick between the two proposals.

        p(sub) = sigmoid((score_sub - score_con) / T). Hard selection
        preserves the integrity of each proposal (no destructive averaging
        of two good points in x-space). Annealed T: exploratory early,
        greedy late.
        """
        T = self.temperature
        g = float(self.sigmoid((score_sub - score_con) / max(T, 1e-6)))
        self.t += 1
        return (x_sub if rng.random() < g else x_con), g

    def blend(self, x_con, x_sub, score_con, score_sub, rng):
        """Backward-compatible alias — hard selection (see select)."""
        return self.select(x_con, x_sub, score_con, score_sub, rng)


# -------------------------------------------------------------
# 5. Bicameral ORCHESTRATOR: Conscious + Subconscious loop
# -------------------------------------------------------------
class BicameralV1:
    """Subconscious-search optimizer tying all upgraded phases together."""

    def __init__(self, dim, lb, ub, latent_dim=None, seed=0,
                 sigma_init=0.2, n_elite=50, beta=1.5,
                 n_init=None, latent_every=10, n_latent_cand=128,
                 n_con_cand=8, sub_noise=0.7, mem_max=512):
        self.dim = dim
        self.lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
        self.ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()
        self.latent_dim = latent_dim or min(dim, 8)
        self.rng = np.random.default_rng(seed)
        self.latent = FastSupervisedLatentSpace(self.latent_dim, n_elite=n_elite)
        # scale sigma_init to domain width
        width = (self.ub - self.lb).mean()
        self._sigma_init = sigma_init * width / 2.0
        self.nes = FastSeparableNES(dim, self.lb, self.ub,
                                    sigma_init=self._sigma_init)
        self.surrogate = FastMaternSurrogate(self.latent_dim)
        self.gate = SigmoidGate()
        self.beta = beta
        self.n_init = n_init or max(2 * dim, 20)
        self.latent_every = latent_every
        self.n_latent_cand = n_latent_cand
        self.n_con_cand = n_con_cand
        self.sub_noise = sub_noise
        self.mem_max = mem_max

    def _random_point(self):
        return self.rng.uniform(self.lb, self.ub)

    def _memory_matrices(self, archive):
        """Elite + recent memory subsample, vectorized into latent coords."""
        n = len(archive)
        if n > self.mem_max:
            order = np.argsort([y for _, y in archive])
            n_half = self.mem_max // 2
            elite_idx = order[:n_half]
            recent_idx = np.arange(n - n_half, n)
            idx = np.unique(np.concatenate([elite_idx, recent_idx]))
        else:
            idx = np.arange(n)
        X = np.array([archive[i][0] for i in idx])
        Y = np.array([archive[i][1] for i in idx], dtype=float)
        C = self.latent.phi(X)  # vectorized: (M, d)
        Y_norm = (Y - Y.mean()) / (Y.std() + 1e-8)
        return C, Y_norm

    def _conscious_candidates(self, best_x):
        """NES samples (no true eval) for surrogate pre-filtering."""
        xs, zs = [], []
        for _ in range(self.n_con_cand):
            x_mut, z = self.nes.step(best_x, self.rng)
            xs.append(x_mut)
            zs.append(z)
        return np.array(xs), np.array(zs)

    def subconscious_proposal(self, best_x, C_mem, Y_mem, include_global=False,
                             global_frac=0.25):
        """Sample in latent space, score with surrogate, decode best.

        Local Gaussian around incumbent (exploit); optionally mixed with
        uniform samples over the memory bounding box (global escape routes).
        Global candidates are only included on epsilon-global iterations or
        right after a restart — otherwise their exploration bonus would
        drown out exploitation and stall convergence.
        """
        c_best = self.latent.phi(best_x)
        if include_global:
            n_global = int(self.n_latent_cand * global_frac)
            n_local = self.n_latent_cand - n_global
            lo, hi = C_mem.min(0), C_mem.max(0)
            pad = (hi - lo) * 0.2 + 1e-6
            C_global = self.rng.uniform(lo - pad, hi + pad,
                                        size=(n_global, self.latent_dim))
            C_local = c_best[None, :] + self.rng.standard_normal(
                (n_local, self.latent_dim)) * self.sub_noise
            C_cand = np.vstack([C_local, C_global])
        else:
            C_cand = c_best[None, :] + self.rng.standard_normal(
                (self.n_latent_cand, self.latent_dim)) * self.sub_noise
        scores = self.surrogate.evaluate_acquisition(
            C_cand, C_mem, Y_mem, beta=self.beta)
        j = int(np.argmax(scores))
        x_sub = self.latent.psi(C_cand[j], self.lb, self.ub)
        return x_sub, float(scores[j])

    def optimize(self, func, max_evals, verbose=False, patience=300,
                 min_rel_improve=1e-3):
        archive = []
        # --- init ---
        best_x, best_y = None, np.inf
        for _ in range(self.n_init):
            x = self._random_point()
            y = float(func(x))
            archive.append((x, y))
            if y < best_y:
                best_x, best_y = x.copy(), y
        init_best = best_y
        n_evals = self.n_init
        history = [best_y]

        it = 0
        gate_sub_picks = 0
        window_start = 0
        window_best = best_y
        restarts = 0
        force_global = False
        eps_global = 0.05
        while n_evals < max_evals:
            # refresh latent model periodically
            if it % self.latent_every == 0:
                self.latent.update(archive)

            if self.latent.ready():
                # Build shared surrogate memory once per iteration
                C_mem, Y_mem = self._memory_matrices(archive)
                # Conscious: best-of-K NES samples by surrogate (unbiased:
                # both sides are max-over-samples, so gate scores compare fairly)
                X_con, Z_con = self._conscious_candidates(best_x)
                s_con_all = self.surrogate.evaluate_acquisition(
                    self.latent.phi(X_con), C_mem, Y_mem, beta=self.beta)
                i = int(np.argmax(s_con_all))
                x_con, z_con, score_con = X_con[i], Z_con[i], float(s_con_all[i])
                # Subconscious: best latent sample by surrogate
                x_sub, score_sub = self.subconscious_proposal(
                    best_x, C_mem, Y_mem)
                # Annealed sigmoid gate decides which single point to evaluate
                use_sub = self.rng.random() < self.gate.sigmoid(
                    (score_sub - score_con) / max(self.gate.temperature, 1e-6))
                self.gate.t += 1
                if use_sub:
                    x_new, z_new, from_sub = x_sub, None, True
                    gate_sub_picks += 1
                else:
                    x_new, z_new, from_sub = x_con, z_con, False
            else:
                # latent not ready yet: pure conscious step
                x_new, z_new = self.nes.step(best_x, self.rng)
                from_sub = False

            y_new = float(func(x_new))
            n_evals += 1
            archive.append((x_new, y_new))
            success = y_new < best_y
            if success:
                best_x, best_y = x_new.copy(), y_new
            # only adapt NES on its own steps — never on subconscious evals
            if not from_sub:
                self.nes.adapt(z_new, success)
            # stagnation restart: relative improvement over the window too
            # small -> re-open exploration (sigma + annealing reboot)
            if it + 1 - window_start >= patience:
                denom = max(abs(window_best), 1e-12)
                rel_improve = (window_best - best_y) / denom
                if rel_improve < min_rel_improve and best_y > 1e-12:
                    self.nes.sigma[:] = self._sigma_init
                    self.gate.t = 0  # annealing reboot: explore again
                    self.latent.update(archive)
                    force_global = True  # one guided global leap next iter
                    restarts += 1
                window_start = it + 1
                window_best = best_y
            history.append(best_y)
            it += 1
            if verbose and (it % 200 == 0 or n_evals >= max_evals):
                print(f"  eval {n_evals}/{max_evals} best={best_y:.6g} "
                      f"T={self.gate.temperature:.4f} sig~{self.nes.sigma.mean():.4f}")
        return best_x, best_y, {"history": history, "init_best": init_best,
                                "evals": n_evals,
                                "gate_sub_frac": gate_sub_picks / max(it, 1),
                                "restarts": restarts}


# -------------------------------------------------------------
# 6. BENCHMARKS
# -------------------------------------------------------------
def sphere(x):
    return float(np.sum(x ** 2))


def rastrigin(x):
    return float(10.0 * len(x) + np.sum(x ** 2 - 10.0 * np.cos(2 * np.pi * x)))


def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ackley(x):
    d = len(x)
    s1 = np.sum(x ** 2)
    s2 = np.sum(np.cos(2 * np.pi * x))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(s1 / d))
                 - np.exp(s2 / d) + 20.0 + np.e)


def griewank(x):
    s = np.sum(x ** 2) / 4000.0
    p = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(s - p + 1.0)


BENCHMARKS = {
    "sphere":     (sphere, -5.0, 5.0),
    "rastrigin":  (rastrigin, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -5.0, 5.0),
    "ackley":     (ackley, -8.0, 8.0),
    "griewank":   (griewank, -10.0, 10.0),
}

# success thresholds (best_loss <= tol counts as solved)
TOLS = {
    "sphere": 1e-4,
    "rastrigin": 1.0,
    "rosenbrock": 1.0,
    "ackley": 1e-2,
    "griewank": 1e-2,
}


def accuracy_score(best_loss):
    """Optimization accuracy in [0,100]: 100 at global optimum (loss 0)."""
    return 100.0 / (1.0 + max(best_loss, 0.0))


def run_benchmark(dim=10, max_evals=2000, n_runs=5, seed0=0, verbose=True):
    results = {}
    for name, (func, lo, hi) in BENCHMARKS.items():
        losses, accs, imps, times = [], [], [], []
        succ = 0
        if verbose:
            print(f"\n=== {name} (dim={dim}, evals={max_evals}, runs={n_runs}) ===")
        for r in range(n_runs):
            opt = BicameralV1(dim, lo, hi, seed=seed0 + r)
            t0 = time.time()
            _, best_y, info = opt.optimize(func, max_evals)
            dt = time.time() - t0
            acc = accuracy_score(best_y)
            imp = 100.0 * (1.0 - best_y / max(info["init_best"], 1e-12))
            losses.append(best_y)
            accs.append(acc)
            imps.append(imp)
            times.append(dt)
            if best_y <= TOLS[name]:
                succ += 1
            if verbose:
                print(f"  run {r+1}: loss={best_y:.6g} acc={acc:.2f}% "
                      f"improv={imp:.2f}% time={dt:.2f}s")
        results[name] = {
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "loss_min": float(np.min(losses)),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "acc_max": float(np.max(accs)),
            "imp_mean": float(np.mean(imps)),
            "success_rate": succ / n_runs,
            "time_mean": float(np.mean(times)),
        }
        if verbose:
            m = results[name]
            print(f"  -> mean loss {m['loss_mean']:.6g} | mean acc "
                  f"{m['acc_mean']:.2f}% | success {m['success_rate']*100:.0f}%")
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Bicameral Upgraded benchmark")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--max-evals", type=int, default=2000)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    print("Bicameral Upgraded — Natural Metric Latent + Sep-CMA Conscious + Sigmoid Gate")
    t0 = time.time()
    results = run_benchmark(dim=args.dim, max_evals=args.max_evals,
                            n_runs=args.n_runs, seed0=args.seed0)
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")
    print("\n================ SUMMARY (mean accuracy %) ================")
    for name, m in results.items():
        print(f"{name:10s} acc={m['acc_mean']:6.2f}%±{m['acc_std']:.2f} "
              f"(best run {m['acc_max']:.2f}%) loss={m['loss_mean']:.4g} "
              f"success={m['success_rate']*100:.0f}%")
    print("===========================================================")


if __name__ == "__main__":
    main()
