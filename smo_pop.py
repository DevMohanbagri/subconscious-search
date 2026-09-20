#!/usr/bin/env python3
"""
SMO-Pop (v2): Full-Covariance Conscious + Population loop.

Upgrades over smo_upgraded.py (v1):
  #1 Conscious: FastFullCMA — textbook full-covariance CMA-ES core
     (weighted recombination, evolution-path cumulation, rank-one + rank-mu
     covariance update, CSA step-size control). Diagonal-only v1 could not
     follow Rosenbrock's curved valley; full covariance fixes that.
  #2 Population: generational (mu/lambda) loop. Each generation evaluates
     lambda CMA offspring PLUS k surrogate-selected subconscious candidates
     (k set adaptively by the annealed sigmoid gate). Stagnation triggers an
     IPOP-style restart: population doubles, sigma resets, mean recenters on
     the global best. No more single-incumbent trap.

All true function evals (conscious + subconscious) count toward max_evals,
so comparisons against other optimizers stay budget-fair.
"""

import time
import numpy as np

from smo_upgraded import (
    FastSupervisedLatentSpace,
    FastMaternSurrogate,
    SigmoidGate,
    BENCHMARKS,
    TOLS,
    accuracy_score,
)


# -------------------------------------------------------------
# 1. FULL-COVARIANCE CMA-ES CONSCIOUS CORE (Hansen-style)
# -------------------------------------------------------------
class FastFullCMA:
    """Compact full CMA-ES core (Hansen-style) incl. Active (negative) update.

    Weighted recombination + evolution-path cumulation + rank-one + rank-mu
    + CSA step-size + active negative weights for the worst offspring with
    Mahalanobis-length safeguard (mirrors pycma's update math, including
    zero-decay negative-weight finalization).
    """

    def __init__(self, dim, lb, ub, m_init, sigma_init, lambda_=None):
        self.N = dim
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        width = float(np.mean(self.ub - self.lb))
        self.sigma_init = sigma_init
        self.sigma_max = width
        self.set_population(lambda_ or (4 + int(3 * np.log(dim))))
        self.m = np.array(m_init, dtype=float).copy()
        self.sigma = sigma_init
        self.C = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.gen = 0

    def set_population(self, lambda_):
        N = self.N
        self.lambda_ = int(lambda_)
        self.mu = max(1, self.lambda_ // 2)
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        w = w / w.sum()
        self.w = w
        self.mueff = 1.0 / np.sum(w ** 2)
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff)
                       / ((N + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (N + 1)) - 1) \
            + self.cs
        self.chiN = np.sqrt(N) * (1 - 1 / (4 * N) - 1 / (21 * N * N))
        self._finalize_active_weights()

    def _finalize_active_weights(self):
        """Negative weights over all-lambda rank weights (pycma-style).

        Raw: w'_i = ln(mu+0.5) - ln(i+1) for i = 0..lambda-1. Positives
        sum to 1; negatives are scaled so that, in order: (1) zero decay
        c1 + cmu*sum(w) = 0, i.e. sum|w^-| = 1 + c1/cmu; (2) positive
        definiteness sum|w^-| <= (1-c1-cmu)/(cmu*N); (3) learning-rate
        limit sum|w^-| <= 1 + 2*mueffminus/(mueff+2). (2),(3) are no-ops
        at default population size.
        """
        N = self.N
        raw = np.log(self.mu + 0.5) - np.log(np.arange(1, self.lambda_ + 1))
        pos = raw[:self.mu]
        pos = pos / pos.sum()
        neg = raw[self.mu:] / np.abs(raw[self.mu:]).sum()  # sum = -1
        # (1) zero decay
        S = 1.0 + self.c1 / self.cmu
        # (2) positive definiteness
        S = min(S, (1 - self.c1 - self.cmu) / self.cmu / N)
        neg = neg * S
        # (3) learning-rate limit (one pass with live mueffminus, as pycma)
        mueffminus = 1.0 / np.sum(neg ** 2)
        S_lim = 1.0 + 2 * mueffminus / (self.mueff + 2)
        if S > S_lim:
            neg = neg * (S_lim / S)
            S = S_lim
        self.w = pos
        self.w_all = np.concatenate([pos, neg])
        self.sum_w_all = float(self.w_all.sum())

    def sample(self, rng, k=None):
        k = k or self.lambda_
        z = rng.standard_normal((k, self.N))
        y = (z * self.D) @ self.B.T          # y ~ N(0, C)
        x = self.m[None, :] + self.sigma * y
        return np.clip(x, self.lb, self.ub)

    def update(self, X_sorted):
        """One CMA update. X_sorted: (lambda_, N) offspring, best-first."""
        N, w, mu = self.N, self.w, self.mu
        m_old = self.m.copy()
        X_mu = X_sorted[:mu]
        self.m = w @ X_mu                    # weighted recombination
        y_w = (self.m - m_old) / self.sigma

        # step-size path (isotropic C^(-1/2) coordinates)
        invsqrt_y = self.B @ ((self.B.T @ y_w) / self.D)
        self.ps = (1 - self.cs) * self.ps \
            + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrt_y
        norm_ps = np.linalg.norm(self.ps)
        hsig = norm_ps / np.sqrt(max(1e-32, 1 - (1 - self.cs) ** (2 * (self.gen + 1)))) \
            < (1.4 + 2 / (N + 1)) * self.chiN

        # covariance path (anisotropic)
        self.pc = (1 - self.cc) * self.pc \
            + (1.0 if hsig else 0.0) * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w

        # active rank-one + rank-mu (+negative) covariance update.
        # Decay uses raw finalized weights; rank terms use Mahalanobis-
        # safeguarded negatives (pycma sampler.update order). With zero
        # negatives this reduces exactly to the standard update.
        Y_all = (X_sorted - m_old) / self.sigma
        w_eff = self.w_all.copy()
        for k in range(mu, self.lambda_):
            if w_eff[k] < 0:
                z = self.B @ ((self.B.T @ Y_all[k]) / self.D)
                w_eff[k] *= N / (float(np.linalg.norm(z)) + 1e-9) ** 2
        delta = (0.0 if hsig else 1.0) * self.cc * (2 - self.cc)
        C = (1 + self.c1 * delta - self.c1 - self.cmu * self.sum_w_all) * self.C \
            + self.c1 * np.outer(self.pc, self.pc) \
            + self.cmu * ((Y_all * w_eff[:, None]).T @ Y_all)
        self.C = (C + C.T) / 2

        # CSA step-size update
        self.sigma *= np.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1))
        self.sigma = float(np.clip(self.sigma, 1e-12, self.sigma_max))

        # eigendecomposition C = B diag(D^2) B^T
        eigvals, B = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-20)
        idx = np.argsort(eigvals)[::-1]
        self.D = np.sqrt(eigvals[idx])
        self.B = B[:, idx]
        self.gen += 1

    def recenter(self, x_best):
        """Adopt a better point found elsewhere (e.g. subconscious basin hop)."""
        self.m = np.array(x_best, dtype=float).copy()
        self.pc[:] = 0.0
        self.ps[:] = 0.0

    def restart(self, new_lambda):
        """IPOP-style restart: bigger population, fresh distribution."""
        self.set_population(new_lambda)
        self.sigma = self.sigma_init
        self.C = np.eye(self.N)
        self.pc[:] = 0.0
        self.ps[:] = 0.0
        self.B = np.eye(self.N)
        self.D = np.ones(self.N)
        self.gen = 0


# -------------------------------------------------------------
# 2. SMO-POP ORCHESTRATOR: population loop + gated subconscious
# -------------------------------------------------------------
class SMOPop:
    """Generational SMO: lambda CMA evals + k gated subconscious evals/gen."""

    def __init__(self, dim, lb, ub, latent_dim=None, seed=0, sigma_init=None,
                 n_elite=100, beta=1.5, n_init=None, latent_every_gen=5,
                 n_latent_cand=256, sub_noise=0.7, mem_max=512,
                 lambda_=None, lambda_max=64):
        self.dim = dim
        self.lb = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
        self.ub = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()
        self.latent_dim = latent_dim or min(dim, 8)
        self.rng = np.random.default_rng(seed)
        self.latent = FastSupervisedLatentSpace(self.latent_dim, n_elite=n_elite)
        self.surrogate = FastMaternSurrogate(self.latent_dim)
        self.gate = SigmoidGate(T0=1.0, T_min=0.05, decay=0.99)
        self.beta = beta
        width = float(np.mean(self.ub - self.lb))
        self.sigma_init = sigma_init or (width / 4.0)
        self.n_init = n_init or max(2 * dim, 20)
        self.latent_every_gen = latent_every_gen
        self.n_latent_cand = n_latent_cand
        self.sub_noise = sub_noise
        self.mem_max = mem_max
        self.lambda_0 = lambda_
        self.lambda_max = lambda_max

    def _random_point(self):
        return self.rng.uniform(self.lb, self.ub)

    def _memory_matrices(self, archive):
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
        C = self.latent.phi(X)
        Y_norm = (Y - Y.mean()) / (Y.std() + 1e-8)
        return C, Y_norm

    def _subconscious_pool(self, best_x, C_mem, Y_mem, global_frac=0.25):
        """Latent candidate pool: local Gaussian + uniform global mixture."""
        c_best = self.latent.phi(best_x)
        n_global = int(self.n_latent_cand * global_frac)
        n_local = self.n_latent_cand - n_global
        C_local = c_best[None, :] + self.rng.standard_normal(
            (n_local, self.latent_dim)) * self.sub_noise
        lo, hi = C_mem.min(0), C_mem.max(0)
        pad = (hi - lo) * 0.2 + 1e-6
        C_global = self.rng.uniform(lo - pad, hi + pad,
                                    size=(n_global, self.latent_dim))
        C_cand = np.vstack([C_local, C_global])
        scores = self.surrogate.evaluate_acquisition(
            C_cand, C_mem, Y_mem, beta=self.beta)
        return C_cand, scores

    def optimize(self, func, max_evals, verbose=False, patience_gens=30,
                 min_rel_improve=1e-3):
        rng = self.rng
        archive = []
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

        cma = FastFullCMA(self.dim, self.lb, self.ub, best_x,
                          self.sigma_init, lambda_=self.lambda_0)
        gen = 0
        restarts = 0
        sub_evals = 0
        window_start = 0  # generation-based (see restart block)
        window_best = best_y
        credit_sub = 1e-3  # decayed improvement credit per stream
        credit_con = 1e-3
        credit_decay = 0.97

        while n_evals + cma.lambda_ <= max_evals:
            if gen % self.latent_every_gen == 0:
                self.latent.update(archive)

            # --- conscious: lambda CMA offspring ---
            X_con = cma.sample(rng)

            # --- subconscious: surrogate-ranked latent pool ---
            # (surrogate scores SELECT which candidates; the gate below
            #  decides HOW MANY via stream credits, not raw scores)
            C_mem, Y_mem = self._memory_matrices(archive)
            C_cand, s_sub = self._subconscious_pool(best_x, C_mem, Y_mem)

            # --- credit-assignment gate: allocation follows production ---
            # Raw acquisition scores cannot see the information value of CMA
            # offspring (they train the covariance model), so a score gate
            # always over-allocates to subconscious look-alikes of the best.
            # Instead each stream earns future allocation from the relative
            # improvements it recently produced (bandit-style, decayed).
            # Annealed T: soft/exploratory early, greedy late.
            p_sub = float(self.gate.sigmoid(
                (np.log(credit_sub + 1e-12) - np.log(credit_con + 1e-12))
                / max(self.gate.temperature, 1e-6)))
            self.gate.t += 1
            progress = n_evals / max_evals
            k_max = max(1, round((cma.lambda_ // 2) * (1 - 0.8 * progress)))
            k_min = 1 if progress < 0.3 else 0
            k = int(np.clip(round(cma.lambda_ * p_sub), k_min, k_max))
            k = min(k, max_evals - n_evals - cma.lambda_)
            if k > 0:
                topk = np.argsort(s_sub)[-k:][::-1]
                X_sub = np.array([self.latent.psi(C_cand[j], self.lb, self.ub)
                                  for j in topk])
            else:
                X_sub = np.zeros((0, self.dim))

            # --- evaluate both pools (all counted toward budget) ---
            Y_con = np.array([float(func(x)) for x in X_con])
            Y_sub = np.array([float(func(x)) for x in X_sub])
            for x, y in zip(X_con, Y_con):
                archive.append((x, y))
            for x, y in zip(X_sub, Y_sub):
                archive.append((x, y))
            n_evals += cma.lambda_ + k
            sub_evals += k

            # --- textbook CMA update on its own offspring ---
            order = np.argsort(Y_con)
            cma.update(X_con[order])

            # --- elitist tracking (CMA mean stays independent!) ---
            # Never recenter the CMA mean mid-run: grafting foreign points
            # into m while keeping C/paths destroys covariance learning
            # (verified: it collapsed rosenbrock 2.7e-05 -> 2.5). The CMA
            # stream and subconscious stream only share the global best;
            # IPOP restarts (fresh C + zeroed paths) are the sole,
            # consistent point where the mean adopts the global best.
            j_con = int(np.argmin(Y_con))
            gen_best, gen_best_x = Y_con[j_con], X_con[j_con]
            sub_won = False
            if k > 0:
                j_sub = int(np.argmin(Y_sub))
                if Y_sub[j_sub] < gen_best:
                    gen_best, gen_best_x = Y_sub[j_sub], X_sub[j_sub]
                    sub_won = True
            if gen_best < best_y:
                rel = (best_y - gen_best) / max(abs(best_y), 1e-12)
                if sub_won:
                    credit_sub += rel
                else:
                    credit_con += rel
                best_y = float(gen_best)
                best_x = gen_best_x.copy()
            credit_sub *= credit_decay
            credit_con *= credit_decay
            history.append(best_y)

            # --- IPOP stagnation restart (GENERATION-based patience) ---
            # Eval-based patience + growing lambda is pathological: at
            # lambda=64 a 300-eval window is <5 generations, so CMA can
            # never adapt before being restarted again. Generations give
            # every population size a fair adaptation window.
            if gen + 1 - window_start >= patience_gens:
                denom = max(abs(window_best), 1e-12)
                rel_improve = (window_best - best_y) / denom
                if rel_improve < min_rel_improve and best_y > 1e-12:
                    cma.restart(min(cma.lambda_ * 2, self.lambda_max))
                    cma.recenter(best_x)
                    self.gate.t = 0
                    self.latent.update(archive)
                    restarts += 1
                window_start = gen + 1
                window_best = best_y

            gen += 1
            if verbose and (gen % 50 == 0 or n_evals + cma.lambda_ > max_evals):
                print(f"  gen {gen} eval {n_evals}/{max_evals} best={best_y:.6g} "
                      f"lam={cma.lambda_} k={k} T={self.gate.temperature:.3f} "
                      f"sig={cma.sigma:.4f}")

        return best_x, best_y, {"history": history, "init_best": init_best,
                                "evals": n_evals, "gens": gen,
                                "restarts": restarts,
                                "sub_frac": sub_evals / max(n_evals, 1)}


# -------------------------------------------------------------
# 3. BENCHMARK (same protocol as v1)
# -------------------------------------------------------------
def run_benchmark(dim=10, max_evals=5000, n_runs=5, seed0=0, verbose=True):
    results = {}
    for name, (func, lo, hi) in BENCHMARKS.items():
        losses, accs, imps, times = [], [], [], []
        succ = 0
        if verbose:
            print(f"\n=== {name} (dim={dim}, evals={max_evals}, runs={n_runs}) ===")
        for r in range(n_runs):
            opt = SMOPop(dim, lo, hi, seed=seed0 + r)
            t0 = time.time()
            _, best_y, info = opt.optimize(func, max_evals)
            dt = time.time() - t0
            from smo_upgraded import accuracy_score as acc_fn
            acc = acc_fn(best_y)
            imp = 100.0 * (1.0 - best_y / max(info["init_best"], 1e-12))
            losses.append(best_y)
            accs.append(acc)
            imps.append(imp)
            times.append(dt)
            if best_y <= TOLS[name]:
                succ += 1
            if verbose:
                print(f"  run {r+1}: loss={best_y:.6g} acc={acc:.2f}% "
                      f"improv={imp:.2f}% time={dt:.2f}s "
                      f"(gens={info['gens']} restarts={info['restarts']} "
                      f"sub={info['sub_frac']:.2f})")
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
    ap = argparse.ArgumentParser(description="SMO-Pop (v2) benchmark")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--max-evals", type=int, default=5000)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    print("SMO-Pop v2 — Full-Covariance Conscious + Population + IPOP restarts")
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
