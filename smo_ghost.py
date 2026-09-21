#!/usr/bin/env python3
"""SMO-Ghost (v4): subconscious-inspired memory architecture.

Implements ideas #8-#14 on top of SMO-Pop v3 (subclass; v3 loop untouched
except no-op hooks):

  #8  Ghost Landscape   - memory of explored regions; good traces attract,
                                trapped regions repel (kernel-weighted bonus).
  #9  Conflict Search   - impressions that disagree (close in latent space,
                                far in value) spawn boundary probes.
  #10 Latent Vector Memory - impressions stored as latent codes (re-encoded
                                whenever the latent frame refreshes).
  #11 Salience/Surprise - unexpected results (|z| vs memory) get stronger
                                initial memory traces.
  #12 Adaptive Memory Decay - traces fade per generation; useful ones
                                (dream parents, probe hits) are refreshed.
  #13 Dreaming          - I_new = a*I_A + (1-a)*I_B + eps in latent space,
                                parents sampled by memory strength.
  #14 Uncertainty-driven exploration - sampling spread adapts to the
                                unfamiliarity of the incumbent region.

Design notes:
- Raw-space X is stored; latent codes are a cache refreshed on every
  frame update (frames rotate, so caching codes alone would go stale).
- With n_dream=n_conflict=ghost_w=0 the run is BIT-IDENTICAL to v3
  (no RNG consumed by disabled paths) - verified in testing.
"""

import time
import numpy as np
from scipy.spatial.distance import cdist

from smo_pop import SMOPop
from smo_upgraded import BENCHMARKS, TOLS, accuracy_score


class GhostMemory:
    """Latent-space impression archive (#8-#14)."""

    def __init__(self, latent_dim, cap=512, decay=0.99, l_scale=1.0,
                 dream_noise=0.35, conflict_noise=0.3, refresh_gain=0.5,
                 max_strength=5.0, salience_cap=3.0, conflict_every=5,
                 top_pairs=8):
        self.d = latent_dim
        self.cap = cap
        self.decay = decay
        self.l_scale = l_scale
        self.dream_noise = dream_noise
        self.conflict_noise = conflict_noise
        self.refresh_gain = refresh_gain
        self.max_strength = max_strength
        self.salience_cap = salience_cap
        self.conflict_every = conflict_every
        self.top_pairs = top_pairs
        self.X = []          # raw-space impressions
        self.Y = []          # their losses
        self.strength = []   # memory strengths
        self.uses = []       # usefulness counts
        self.C = np.zeros((0, latent_dim))  # latent codes cache
        self._pairs = []
        self._gen = 0

    @property
    def n(self):
        return len(self.Y)

    # -- #10 latent codes (cache; frames rotate so re-encode on update) --
    def refresh_codes(self, latent):
        if self.n:
            self.C = np.asarray(latent.phi(np.array(self.X)), dtype=float)

    # -- #11 salience: surprise vs strength-weighted memory stats --
    def add_batch(self, Xb, Yb):
        Xb = np.asarray(Xb, dtype=float)
        Yb = np.asarray(Yb, dtype=float).ravel()
        if self.n >= 2:
            w = np.asarray(self.strength, dtype=float)
            w = w / (w.sum() + 1e-12)
            mu = float(w @ np.asarray(self.Y))
            sd = float(np.sqrt(max(w @ (np.asarray(self.Y) - mu) ** 2, 1e-24)))
        else:
            mu = float(Yb.mean())
            sd = float(Yb.std()) if len(Yb) > 1 else 1.0
            sd = max(sd, 1e-12)
        for x, y in zip(Xb, Yb):
            s0 = 1.0 + min(abs(float(y) - mu) / sd, self.salience_cap)
            self.X.append(x.copy())
            self.Y.append(float(y))
            self.strength.append(s0)
            self.uses.append(0)
        self._prune()

    def _prune(self):
        while len(self.Y) > self.cap:
            j = int(np.argmin(self.strength))
            del self.X[j], self.Y[j], self.strength[j], self.uses[j]
        # NOTE: C cache re-synced by caller via refresh_codes()

    # -- #12 decay (+ periodic conflict mining) --
    def end_of_generation(self):
        self._gen += 1
        self.strength = [s * self.decay for s in self.strength]
        if self.n >= 8 and self._gen % self.conflict_every == 0:
            self._update_conflicts()
        self._prune()

    # -- kernel helpers --
    def _rbf(self, Cq):
        if self.n == 0:
            return np.zeros((len(Cq), 0))
        d2 = cdist(np.asarray(Cq, dtype=float), self.C, metric="sqeuclidean")
        return np.exp(-d2 / (2 * self.l_scale ** 2))

    def _goodness(self):
        """Rank-based goodness in [0,1]; best impression = 1."""
        Y = np.asarray(self.Y, dtype=float)
        if self.n == 1:
            return np.ones(1)
        order = np.argsort(Y)
        g = np.empty(self.n)
        g[order] = 1.0 - np.arange(self.n) / (self.n - 1)
        return g

    # -- #8 ghost value: attract good traces, repel trapped ones --
    def ghost_value(self, Cq):
        """Kernel-weighted goodness in [0,1]; 0.5 when memory is empty."""
        if self.n == 0:
            return np.full(len(np.asarray(Cq)), 0.5)
        K = self._rbf(Cq)
        w = K * np.asarray(self.strength)[None, :]
        g = self._goodness()
        num = w @ g
        den = w.sum(axis=1)
        out = np.full(len(num), 0.5)
        m = den > 1e-12
        out[m] = num[m] / den[m]
        return out

    # -- #14 unfamiliarity in [0,1] from strength-weighted density --
    def unfamiliarity(self, c):
        if self.n == 0:
            return 1.0
        k = self._rbf(np.asarray(c, dtype=float)[None, :])[0]
        d = float(k @ np.asarray(self.strength))
        return 1.0 / (1.0 + d)

    # -- #12 usefulness refresh --
    def refresh(self, idxs, gain=None):
        gain = self.refresh_gain if gain is None else gain
        for i in idxs:
            i = int(i)
            if 0 <= i < self.n:
                self.strength[i] = min(self.strength[i] + gain,
                                       self.max_strength)
                self.uses[i] += 1

    # -- #9 conflict mining: similar situation, disagreeing value --
    def _update_conflicts(self):
        g = self._goodness()
        d2 = cdist(self.C, self.C, metric="sqeuclidean")
        sim = np.exp(-d2 / (2 * self.l_scale ** 2))
        gap = np.abs(g[:, None] - g[None, :])
        score = sim * gap
        np.fill_diagonal(score, 0.0)
        # greedy disjoint pairs, most disagreement first
        order = np.dstack(np.unravel_index(np.argsort(-score.ravel()),
                                           score.shape))[0]
        used, pairs = set(), []
        for i, j in order:
            i, j = int(i), int(j)
            if i in used or j in used:
                continue
            if score[i, j] <= 1e-9:
                break
            pairs.append((i, j))
            used.add(i)
            used.add(j)
            if len(pairs) >= self.top_pairs:
                break
        self._pairs = pairs

    # -- #13 dreaming: recombine strong impressions --
    # NOTE: convex blending (a*A + (1-a)*B) was tried first and HURT on
    # multimodal landscapes (rastrigin 50% -> 25%): midpoints of two good
    # basins decode to the barrier between them, and the smoothing
    # surrogate systematically over-scores those midpoints, so barrier
    # tops get selected for evaluation. Mask crossover instead combines
    # PIECES (per-coordinate mask): offspring inherit whole coordinates
    # from each parent, preserving basin membership per coordinate -
    # closer to the idea's stated goal of "novel candidate basins".
    def dream(self, rng, n):
        if self.n < 2 or n <= 0:
            return np.zeros((0, self.d)), []
        w = np.asarray(self.strength, dtype=float)
        w = w / (w.sum() + 1e-12)
        ia = rng.choice(self.n, size=n, p=w)
        ib = rng.choice(self.n, size=n, p=w)
        mask = rng.random((n, self.d)) < 0.5
        C_new = np.where(mask, self.C[ia], self.C[ib]) \
            + rng.standard_normal((n, self.d)) * self.dream_noise
        return C_new, list(ia) + list(ib)

    def conflict_probes(self, rng, n):
        # NOTE: landing ON the midpoint was tried first: it evaluates
        # barrier tops (close + disagreeing value = steep slope between
        # basins). Probes now jump THROUGH the boundary to either side,
        # alternating, where the unexplored basin may lie.
        if not self._pairs or n <= 0:
            return np.zeros((0, self.d)), []
        out, used = [], []
        for t in range(n):
            i, j = self._pairs[t % len(self._pairs)]
            side = 1.0 if (t // max(len(self._pairs), 1)) % 2 == 0 else -1.0
            diff = self.C[j] - self.C[i]
            probe = 0.5 * (self.C[i] + self.C[j]) + side * 0.75 * diff \
                + rng.standard_normal(self.d) * self.conflict_noise
            out.append(probe)
            used.extend([i, j])
        return np.array(out), used


class SMOGhost(SMOPop):
    """SMO-Pop v3 + GhostMemory subconscious (ideas #8-#14)."""

    def __init__(self, *args, ghost_w=0.5, n_dream=64, n_conflict=32,
                 ghost_cap=512, **kwargs):
        super().__init__(*args, **kwargs)
        self.ghost = GhostMemory(self.latent_dim, cap=ghost_cap)
        self.ghost_w = ghost_w
        self.n_dream = n_dream
        self.n_conflict = n_conflict

    # -- hooks --
    def _hook_latent_updated(self):
        self.ghost.refresh_codes(self.latent)

    def _hook_new_evals(self, X_all, Y_all):
        self.ghost.add_batch(X_all, Y_all)
        if self.latent.ready():
            self.ghost.refresh_codes(self.latent)

    def _hook_end_of_generation(self, gen):
        self.ghost.end_of_generation()

    # -- subconscious pool with dream + conflict + ghost scoring --
    def _subconscious_pool(self, best_x, C_mem, Y_mem, global_frac=0.25):
        rng = self.rng
        ghost_on = (self.n_dream > 0 or self.n_conflict > 0
                    or self.ghost_w > 0)
        c_best = self.latent.phi(best_x)
        if ghost_on and self.ghost.n >= 2:
            # #14: unfamiliar regions get a wider sampling spread
            unfam = self.ghost.unfamiliarity(c_best)
            eff_noise = self.sub_noise * (0.5 + 1.5 * unfam)
        else:
            eff_noise = self.sub_noise
        n_global = int(self.n_latent_cand * global_frac)
        n_local = self.n_latent_cand - n_global
        C_local = c_best[None, :] + rng.standard_normal(
            (n_local, self.latent_dim)) * eff_noise
        lo, hi = C_mem.min(0), C_mem.max(0)
        pad = (hi - lo) * 0.2 + 1e-6
        C_global = rng.uniform(lo - pad, hi + pad,
                               size=(n_global, self.latent_dim))
        parts = [C_local, C_global]
        if self.n_dream > 0 and self.ghost.n >= 2:
            Cd, parents = self.ghost.dream(rng, self.n_dream)
            self.ghost.refresh(parents)  # #12: dreaming is using
            parts.append(Cd)
        if self.n_conflict > 0 and self.ghost.n >= 2:
            Cc, used = self.ghost.conflict_probes(rng, self.n_conflict)
            if len(Cc):
                self.ghost.refresh(used)
                parts.append(Cc)
        C_cand = np.vstack(parts)
        scores = self.surrogate.evaluate_acquisition(
            C_cand, C_mem, Y_mem, beta=self.beta)
        if self.ghost_w > 0 and self.ghost.n >= 2:
            # #8: attract good traces, repel trapped ones
            v = self.ghost.ghost_value(C_cand)
            scores = scores + self.ghost_w * (v - 0.5) * 2.0
        return C_cand, scores


def run_benchmark(dim=10, max_evals=5000, n_runs=5, seed0=0, verbose=True,
                  patience_gens=30, **ghost_kwargs):
    from smo_upgraded import accuracy_score as acc_fn
    results = {}
    for name, (func, lo, hi) in BENCHMARKS.items():
        losses, accs, imps, times = [], [], [], []
        succ = 0
        if verbose:
            print(f"\n=== {name} (dim={dim}, evals={max_evals}, runs={n_runs}) ===",
                  flush=True)
        for r in range(n_runs):
            opt = SMOGhost(dim, lo, hi, seed=seed0 + r, **ghost_kwargs)
            t0 = time.time()
            _, best_y, info = opt.optimize(func, max_evals,
                                           patience_gens=patience_gens)
            dt = time.time() - t0
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
                      f"(ghosts={opt.ghost.n})", flush=True)
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
    ap = argparse.ArgumentParser(description="SMO-Ghost (v4) benchmark")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--max-evals", type=int, default=5000)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--ghost-w", type=float, default=0.5)
    ap.add_argument("--n-dream", type=int, default=64)
    ap.add_argument("--n-conflict", type=int, default=32)
    ap.add_argument("--no-ghost", action="store_true",
                    help="disable all ghost machinery (must match v3 bit-for-bit)")
    args = ap.parse_args()
    kw = {"ghost_w": 0.0, "n_dream": 0, "n_conflict": 0} if args.no_ghost else {
        "ghost_w": args.ghost_w, "n_dream": args.n_dream,
        "n_conflict": args.n_conflict}
    print(f"SMO-Ghost v4 — ideas #8-#14 ({'DISABLED (v3 check)' if args.no_ghost else kw})")
    t0 = time.time()
    results = run_benchmark(dim=args.dim, max_evals=args.max_evals,
                            n_runs=args.n_runs, seed0=args.seed0, **kw)
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")
    print("\n================ SUMMARY (mean accuracy %) ================")
    for name, m in results.items():
        print(f"{name:10s} acc={m['acc_mean']:6.2f}%±{m['acc_std']:.2f} "
              f"(best run {m['acc_max']:.2f}%) loss={m['loss_mean']:.4g} "
              f"success={m['success_rate']*100:.0f}%")
    print("===========================================================")


if __name__ == "__main__":
    main()
