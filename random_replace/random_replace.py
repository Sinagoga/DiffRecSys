import numpy as np


def random_replace_corrupt(sids, t, codebook_size, rng=None):
    """Forward corruption q(x_t | x_0) for the uniform kernel.

    Args:
        sids: (B, L) int array of codebook ids; not mutated.
        t: scalar in [0,1], or array of shape (B,), (B,1), or (B,L) — the
           marginal replacement probability (noise level).
        codebook_size: int K (shared), or array of length L for per-digit sizes;
           replacements are drawn from [0, K).
        rng: optional np.random.Generator for reproducibility.
    Returns:
        corrupted: (B, L) int array, same dtype as sids.
        resampled: (B, L) bool — positions redrawn (a redraw may equal x0).
    """
    rng = np.random.default_rng() if rng is None else rng
    sids = np.asarray(sids)
    if sids.ndim != 2:
        raise ValueError(f"sids must be 2-D (B, L); got {sids.shape}")
    B, L = sids.shape
    t_arr = np.asarray(t, dtype=float)
    if t_arr.ndim == 0:
        t_arr = np.full((B, 1), float(t_arr))
    elif t_arr.shape == (B,):
        t_arr = t_arr.reshape(B, 1)
    elif t_arr.shape not in {(B, 1), (B, L)}:
        raise ValueError(f"t must be scalar, (B,), (B,1) or (B,L); got {t_arr.shape}")
    high = np.asarray(codebook_size)
    if np.any(high < 1):
        raise ValueError("codebook_size must be >= 1")
    if not np.can_cast(np.min_scalar_type(int(high.max()) - 1), sids.dtype):
        raise ValueError(f"sids.dtype {sids.dtype} cannot hold ids up to {int(high.max()) - 1}")
    resampled = rng.random((B, L)) < t_arr
    random_tokens = rng.integers(0, high, size=(B, L), dtype=sids.dtype)
    corrupted = np.where(resampled, random_tokens, sids)
    return corrupted, resampled


def training_pair(sids, codebook_size, rng=None, t_low=0.0, t_high=1.0):
    """Sample a per-item noise level, corrupt, and return the all-position target.

    Returns (corrupted, target, t): target == sids (predict the clean token at
    EVERY position), t has shape (B, 1). Train with cross-entropy over all L
    positions — uniform diffusion has no mask to restrict the loss to.
    """
    rng = np.random.default_rng() if rng is None else rng
    sids = np.asarray(sids)
    t = rng.uniform(t_low, t_high, size=(sids.shape[0], 1))
    corrupted, _ = random_replace_corrupt(sids, t, codebook_size, rng)
    return corrupted, sids.copy(), t


def denoise_step(x_t, x0_pred, t_now, t_next, rng=None):
    """One pragmatic reverse step: keep current tokens with prob t_next/t_now,
    else commit the model's predicted clean token x0_pred. At t_next=0 the whole
    sequence becomes x0_pred. (Approximation of the exact D3PM uniform posterior;
    swap in the closed-form posterior for exactness.)"""
    rng = np.random.default_rng() if rng is None else rng
    x_t, x0_pred = np.asarray(x_t), np.asarray(x0_pred)
    keep_prob = 0.0 if t_now <= 0 else max(0.0, min(1.0, t_next / t_now))
    keep = rng.random(x_t.shape) < keep_prob
    return np.where(keep, x_t, x0_pred)


def generate(predict_fn, shape, codebook_size, n_steps=20, rng=None):
    """Sample SIDs: start from Uniform(codebook), denoise for n_steps.
    predict_fn(x_t, t) -> x0_pred of shape `shape`."""
    rng = np.random.default_rng() if rng is None else rng
    x = rng.integers(0, codebook_size, size=shape)
    ts = np.linspace(1.0, 0.0, n_steps + 1)
    for i in range(n_steps):
        x = denoise_step(x, predict_fn(x, ts[i]), ts[i], ts[i + 1], rng)
    return x


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x0 = rng.integers(0, 512, size=(256, 10), dtype=np.int32)  # project SIDs: L=10, K=512

    c0, _ = random_replace_corrupt(x0, 0.0, 512, rng)
    assert np.array_equal(c0, x0), "t=0 must leave the sequence unchanged"

    c1, r1 = random_replace_corrupt(x0, 1.0, 512, rng)
    assert c1.min() >= 0 and c1.max() < 512, "tokens must stay in [0, K)"
    assert r1.all(), "t=1 must resample every position"
    assert (c1 == x0).mean() < 0.05, "at t=1 only ~1/K survive by chance"

    cm, rm = random_replace_corrupt(x0, 0.3, 512, rng)
    assert abs(rm.mean() - 0.3) < 0.05, "resample rate must track t"

    xt, tgt, t = training_pair(x0, 512, rng)
    assert np.array_equal(tgt, x0) and t.shape == (256, 1), "target is x0 at all positions"

    out = generate(lambda x, _t: np.zeros_like(x), (4, 10), 512, n_steps=10, rng=rng)
    assert out.shape == (4, 10) and out.min() >= 0 and out.max() < 512

    # per-digit codebook sizes (general case) also work
    cpd, _ = random_replace_corrupt(x0[:, :3], 1.0, np.array([512, 256, 128]), rng)
    assert (cpd[:, 1] < 256).all() and (cpd[:, 2] < 128).all(), "per-digit ranges respected"

    print("random_replace self-test: PASSED")
