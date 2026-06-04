"""Disentangle spectral health from task fine-tuning by training two identical
MLPs with vs without wwj.alpha_loss as a regularizer. Tests the Muon-alternative
prediction from Martin's RG theory paper (arXiv 2026): directly penalizing
|alpha - 2| per layer (via wwj.alpha_loss) should drive the trained network
toward Martin's RG-optimal fixed point and improve robustness on held-out
perturbations, without the indirect-orthogonalization machinery Muon needs.

The cohort_sweep / multi_seed_sweep results showed that BrainIAC's brainage
ckpt (mean |alpha-2|=0.12) was more robust than BrainIAC pretrained
(mean |alpha-2|=0.40) -- but the confound is that brainage was also fine-tuned
on the test domain. This experiment removes the confound: same architecture,
same data, same seed, ONLY difference is the alpha_loss regularizer.

Setup (mirrors Martin's Section 7 MLP-on-MNIST proposed experiment):
  - sklearn 8x8 digits (1797 samples; no download needed)
  - 3-hidden-layer MLP, 512-d hidden, ReLU
  - AdamW, lr=1e-3, wd=1e-4
  - 2000 steps, batch=128
  - alpha_loss weight=0.01, target=2.0 (Hill estimator, frac=0.5)

Measurements:
  - per-100-step: train loss, test acc, mean alpha across hidden-layer
    matrices, mean |alpha - 2|
  - final: noise robustness curve under Gaussian noise injection
  - both: same PRNG seed for fair comparison

Usage:
    uv run python benchmarks/alpha_loss_disentangle.py
    uv run python benchmarks/alpha_loss_disentangle.py --steps 4000 --alpha-weight 0.05
"""

from __future__ import annotations

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from wwj.core import _eigvals, _hill_alpha, _walk_matrices
import wwj


class MLP(eqx.Module):
    """Small MLP whose hidden-layer weight matrices are min-dim 50+ so they're
    in wwj.alpha_loss's analysis scope. The 10-d output layer is excluded
    automatically by the min_dim=50 filter."""
    layers: list

    def __init__(self, key, in_dim=64, hidden=512, depth=3, out_dim=10):
        keys = jax.random.split(key, depth + 1)
        sizes = [in_dim] + [hidden] * depth + [out_dim]
        self.layers = [eqx.nn.Linear(sizes[i], sizes[i + 1], key=k)
                       for i, k in enumerate(keys)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


def make_loss_fn(use_alpha_loss: bool, alpha_weight: float):
    def loss_fn(model, x, y):
        logits = jax.vmap(model)(x)
        task = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        if use_alpha_loss:
            reg = wwj.alpha_loss(model, target=2.0, weight=alpha_weight,
                                  min_dim=50, hill_frac=0.5)
            return task + reg
        return task
    return loss_fn


def measure_alpha_stats(model, min_dim=50):
    """Return list of per-layer alpha + (mean, mean_|alpha-2|)."""
    mats = _walk_matrices(model, min_dim=min_dim)
    alphas = [float(_hill_alpha(_eigvals(W))) for _, W in mats]
    if not alphas:
        return [], float("nan"), float("nan")
    a = np.array(alphas)
    return alphas, float(a.mean()), float(np.abs(a - 2.0).mean())


def train_one(seed, X_train, y_train, X_test, y_test, use_alpha_loss,
              alpha_weight, n_steps, batch_size, log_every):
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    model = MLP(init_key, in_dim=X_train.shape[1], hidden=512, depth=3, out_dim=10)

    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    loss_fn = make_loss_fn(use_alpha_loss, alpha_weight)

    @jax.jit
    def step_fn(model, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state,
                                              eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    n_train = X_train.shape[0]
    traj = []
    t0 = time.time()
    for step in range(n_steps):
        key, sk = jax.random.split(key)
        idx = jax.random.choice(sk, n_train, (batch_size,), replace=False)
        model, opt_state, loss = step_fn(model, opt_state, X_train[idx], y_train[idx])

        if step % log_every == 0 or step == n_steps - 1:
            logits = jax.vmap(model)(X_test)
            test_acc = float((logits.argmax(-1) == y_test).mean())
            alphas, mean_a, dist = measure_alpha_stats(model)
            traj.append({"step": step, "loss": float(loss), "test_acc": test_acc,
                         "mean_alpha": mean_a, "alpha_dist_from_2": dist,
                         "alphas": alphas})
    print(f"  trained in {time.time() - t0:.1f}s")
    return model, traj


def noise_sweep(model, X_test, y_test, noise_stds, seed=0):
    """Accuracy under additive Gaussian noise on test inputs."""
    rng = np.random.default_rng(seed)
    accs = []
    for std in noise_stds:
        if std == 0:
            X_p = X_test
        else:
            noise = rng.standard_normal(X_test.shape).astype(np.float32)
            X_p = X_test + std * jnp.asarray(noise)
        logits = jax.vmap(model)(X_p)
        accs.append(float((logits.argmax(-1) == y_test).mean()))
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--alpha-weight", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # sklearn 8x8 digits -- ships with sklearn, no download needed.
    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data.astype(np.float32) / 16.0, digits.target
    n_train = 1500
    X_train = jnp.array(X[:n_train]); y_train = jnp.array(y[:n_train])
    X_test = jnp.array(X[n_train:]); y_test = jnp.array(y[n_train:])
    print(f"Data: {X_train.shape} train / {X_test.shape} test, in_dim={X.shape[1]}, classes=10")

    print(f"\n=== baseline (no alpha_loss) ===")
    model_base, traj_base = train_one(args.seed, X_train, y_train, X_test, y_test,
                                       use_alpha_loss=False, alpha_weight=0.0,
                                       n_steps=args.steps, batch_size=args.batch,
                                       log_every=args.log_every)

    print(f"\n=== +alpha_loss (weight={args.alpha_weight}, target=2.0) ===")
    model_reg, traj_reg = train_one(args.seed, X_train, y_train, X_test, y_test,
                                     use_alpha_loss=True, alpha_weight=args.alpha_weight,
                                     n_steps=args.steps, batch_size=args.batch,
                                     log_every=args.log_every)

    # === Trajectories ===
    print(f"\n{'='*72}")
    print(f"Trajectory (every {args.log_every} steps)")
    print(f"{'='*72}")
    print(f"{'step':>5s}  {'base_loss':>10s} {'base_acc':>9s} {'base_α':>7s} {'base|Δ|':>8s}  "
          f"{'reg_loss':>9s} {'reg_acc':>8s} {'reg_α':>6s} {'reg|Δ|':>7s}")
    for tb, tr in zip(traj_base, traj_reg):
        print(f"{tb['step']:>5d}  {tb['loss']:>10.4f} {tb['test_acc']:>9.3f} "
              f"{tb['mean_alpha']:>7.2f} {tb['alpha_dist_from_2']:>8.2f}  "
              f"{tr['loss']:>9.4f} {tr['test_acc']:>8.3f} "
              f"{tr['mean_alpha']:>6.2f} {tr['alpha_dist_from_2']:>7.2f}")

    # === Per-layer final α ===
    print(f"\n{'='*72}")
    print(f"Final per-layer alpha distribution")
    print(f"{'='*72}")
    print(f"{'layer':>5s}  {'baseline':>10s}  {'+alpha_loss':>13s}  {'Δ':>8s}")
    for i, (a_b, a_r) in enumerate(zip(traj_base[-1]["alphas"], traj_reg[-1]["alphas"])):
        print(f"{i:>5d}  {a_b:>10.3f}  {a_r:>13.3f}  {a_r - a_b:>+8.3f}")

    # === Noise robustness ===
    print(f"\n{'='*72}")
    print(f"Robustness sweep: Gaussian noise on test inputs")
    print(f"{'='*72}")
    noise_stds = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    base_accs = noise_sweep(model_base, X_test, y_test, noise_stds)
    reg_accs = noise_sweep(model_reg, X_test, y_test, noise_stds)
    print(f"{'noise σ':>8s}  {'baseline':>10s}  {'+α_loss':>10s}  {'Δ acc':>8s}")
    for s, ba, ra in zip(noise_stds, base_accs, reg_accs):
        print(f"{s:>8.2f}  {ba:>10.3f}  {ra:>10.3f}  {ra - ba:>+8.3f}")

    # error slope = (1-acc) vs noise stdev
    base_err_slope = float(np.polyfit(noise_stds, [1 - a for a in base_accs], 1)[0])
    reg_err_slope = float(np.polyfit(noise_stds, [1 - a for a in reg_accs], 1)[0])
    print(f"\nError-rate slope (lower = more robust):")
    print(f"  baseline:    {base_err_slope:.4f}  per unit noise σ")
    print(f"  +alpha_loss: {reg_err_slope:.4f}  per unit noise σ")
    print(f"  Δ:           {reg_err_slope - base_err_slope:+.4f}")
    if reg_err_slope < base_err_slope:
        print(f"  ✓ +alpha_loss is MORE ROBUST -- the RG-direct intervention works.")
    else:
        print(f"  ✗ alpha_loss did not improve robustness in this run.")


if __name__ == "__main__":
    main()
