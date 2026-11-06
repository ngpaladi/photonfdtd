"""Gradient-descent loops for chip geometry optimization.

The gradient itself comes from the time-domain adjoint
(:func:`~photonfdtd.make_value_and_grad_params`); this module supplies the
loop around it:

* :class:`Adam` - a dependency-free Adam that follows the optax
  ``GradientTransformation`` protocol (``init`` / ``update``), so any optax
  optimizer or chain (``optax.adam``, ``optax.chain(optax.clip(1), ...)``) can
  be passed in its place.
* :func:`minimize` - projected gradient descent: step, clip to the box
  bounds, repeat. Hyper-parameters such as the projection sharpness come from
  a ``schedule(iteration)`` (e.g. :func:`beta_continuation`) and are passed to
  the compiled gradient as traced values, so the schedule never recompiles.
  An optional ``.npz`` checkpoint is written every iteration and resumed
  transparently.
* :func:`robust` - eroded / nominal / dilated variants of a density design
  (projection thresholds ``eta`` shifted about 0.5), combined into a minimax
  or mean objective, so the optimized device tolerates over- and under-etch.
* :class:`ModeTransmission` / :func:`combine` - S-parameter objectives built
  from the mode-overlap amplitudes of :mod:`photonfdtd.smatrix`.

References
----------
* D. P. Kingma & J. Ba, "Adam: A method for stochastic optimization,"
  ICLR (2015).
* F. Wang, B. Lazarov, O. Sigmund, *Struct. Multidiscip. Optim.* 43, 767
  (2011) - robust (eroded / dilated) formulation.
* A. Hammond et al., "High-performance hybrid time/frequency-domain topology
  optimization for large-scale photonics inverse design," *Opt. Express* 30,
  4467 (2022) - the time-domain adjoint + continuation workflow.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #

class AdamState(NamedTuple):
    count: object
    mu: object
    nu: object


class Adam:
    """Adam (Kingma & Ba) in the optax ``init`` / ``update`` protocol.

    ``update(grads, state, params)`` returns ``(updates, state)``; the new
    parameters are ``params + updates``. Matches ``optax.adam`` step for step.
    """

    def __init__(self, learning_rate: float = 0.01, b1: float = 0.9,
                 b2: float = 0.999, eps: float = 1e-8):
        self.learning_rate = learning_rate
        self.b1, self.b2, self.eps = b1, b2, eps

    def init(self, params):
        import jax
        import jax.numpy as jnp
        z = jax.tree_util.tree_map(jnp.zeros_like, params)
        return AdamState(jnp.zeros([], jnp.int32), z, z)

    def update(self, grads, state, params=None):
        import jax
        import jax.numpy as jnp
        b1, b2 = self.b1, self.b2
        count = state.count + 1
        mu = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g,
                                    state.mu, grads)
        nu = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g,
                                    state.nu, grads)
        c1 = 1 - b1 ** count
        c2 = 1 - b2 ** count
        upd = jax.tree_util.tree_map(
            lambda m, v: -self.learning_rate * (m / c1)
            / (jnp.sqrt(v / c2) + self.eps), mu, nu)
        return upd, AdamState(count, mu, nu)


# --------------------------------------------------------------------------- #
# Schedules and metrics
# --------------------------------------------------------------------------- #

def beta_continuation(beta0: float = 1.0, factor: float = 2.0, every: int = 20,
                      beta_max: float = 64.0, **fixed):
    """Projection-sharpness schedule: ``beta0 * factor**(it // every)``.

    Returns ``schedule(it) -> {'beta': ..., **fixed}`` for :func:`minimize`.
    Starting soft and sharpening lets the optimizer move material freely
    before the design is pushed to binary.
    """
    def schedule(it):
        return dict(beta=float(min(beta0 * factor ** (it // every), beta_max)),
                    **fixed)
    return schedule


def binarization(rho_phys) -> float:
    """``1 - 4 mean(rho (1 - rho))``: 1 for a binary design, 0 for all-0.5."""
    r = np.asarray(rho_phys, float)
    return float(1.0 - 4.0 * np.mean(r * (1.0 - r)))


def robust(vg, etas: Sequence[float] = (0.45, 0.5, 0.55), reduce: str = "max"):
    """Combine eroded / nominal / dilated variants of a density design.

    ``vg(params, eta=..., **aux)`` is a compiled gradient whose coefficient
    builder takes the projection threshold ``eta`` (e.g. from
    :meth:`DesignRegion.ce_builder`); a larger ``eta`` erodes the design
    (less solid), a smaller one dilates it. ``reduce='max'`` returns the
    worst variant's value and gradient (minimax for a minimized loss);
    ``'mean'`` averages them. Each call costs ``len(etas)`` simulations.
    """
    if reduce not in ("max", "mean"):
        raise ValueError("reduce must be 'max' or 'mean'")

    def vg_robust(params, **aux):
        import jax
        res = [vg(params, eta=e, **aux) for e in etas]
        vals = np.array([float(v) for v, _ in res])
        if reduce == "max":
            k = int(np.argmax(vals))
            return res[k][0], res[k][1]
        g = jax.tree_util.tree_map(lambda *gs: sum(gs) / len(gs),
                                   *[g for _, g in res])
        return float(vals.mean()), g
    return vg_robust


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #

@dataclass
class OptResult:
    """``x`` - final parameters; ``history`` - one dict per iteration."""
    x: object
    history: List[Dict] = field(default_factory=list)

    def values(self) -> np.ndarray:
        return np.array([h["value"] for h in self.history])


def _scalar_items(d):
    out = {}
    for k, v in (d or {}).items():
        a = np.asarray(v)
        if a.ndim == 0 and np.isrealobj(a):
            out[k] = float(a)
    return out


def _save_checkpoint(path, it, x, opt_state, history):
    import jax
    leaves = jax.tree_util.tree_leaves(opt_state)
    arrays = {f"s{i}": np.asarray(l) for i, l in enumerate(leaves)}
    tmp = path + ".tmp.npz"
    np.savez(tmp, it=it, x=np.asarray(x), history=json.dumps(history), **arrays)
    os.replace(tmp, path)


def _load_checkpoint(path, opt_state):
    import jax
    import jax.numpy as jnp
    with np.load(path) as z:
        treedef = jax.tree_util.tree_structure(opt_state)
        n = treedef.num_leaves
        leaves = [jnp.asarray(z[f"s{i}"]) for i in range(n)]
        return (int(z["it"]), jnp.asarray(z["x"]),
                jax.tree_util.tree_unflatten(treedef, leaves),
                json.loads(str(z["history"])))


def minimize(value_and_grad, x0, optimizer=None, n_iter: int = 50,
             bounds=(0.0, 1.0), schedule: Optional[Callable] = None,
             callback: Optional[Callable] = None,
             checkpoint: Optional[str] = None, maximize: bool = False,
             verbose: bool = False) -> OptResult:
    """Projected gradient descent on ``value_and_grad``.

    Scale the loss to order one (e.g. normalize by a reference run): Adam's
    ``eps`` swamps the gradient of a loss in raw FDTD field units (~1e-30).

    Parameters
    ----------
    value_and_grad : callable
        ``vg(x, **aux) -> (value, grad)`` (or ``(value, grad, info)`` with a
        dict of scalar diagnostics to log), e.g. from
        :func:`~photonfdtd.make_value_and_grad_params`.
    x0 : array
        Starting parameters (e.g. a density, 0.5 everywhere).
    optimizer : optax-style transformation, optional
        Anything with ``init(params)`` and ``update(grads, state, params)``;
        defaults to ``Adam(0.01)``.
    n_iter : int
        Total iterations (counting any resumed from ``checkpoint``).
    bounds : (lo, hi) or None
        Box constraint applied after every step (``None`` disables).
    schedule : callable, optional
        ``schedule(it) -> dict`` of keyword arguments for ``vg``
        (e.g. :func:`beta_continuation`).
    callback : callable, optional
        ``callback(it, x, record)`` after every step; return ``True`` to stop.
    checkpoint : str, optional
        ``.npz`` path written after each iteration; if it exists the run
        resumes from it (same result as an uninterrupted run).
    maximize : bool
        Ascend instead of descend.
    """
    import jax
    import jax.numpy as jnp

    opt = optimizer if optimizer is not None else Adam(0.01)
    x = jnp.asarray(x0)
    state = opt.init(x)
    history: List[Dict] = []
    start = 0
    if checkpoint and os.path.exists(checkpoint):
        start, x, state, history = _load_checkpoint(checkpoint, state)
    sign = -1.0 if maximize else 1.0

    for it in range(start, n_iter):
        aux = schedule(it) if schedule is not None else {}
        t0 = time.perf_counter()
        res = value_and_grad(x, **aux)
        val, grad = res[0], res[1]
        info = res[2] if len(res) > 2 else {}
        g = jax.tree_util.tree_map(lambda a: sign * jnp.real(a), grad)
        upd, state = opt.update(g, state, x)
        x = x + upd
        if bounds is not None:
            x = jnp.clip(x, bounds[0], bounds[1])
        rec = dict(it=it, value=float(val),
                   grad_norm=float(jnp.sqrt(jnp.sum(jnp.abs(grad) ** 2))),
                   time=time.perf_counter() - t0)
        rec.update(_scalar_items(aux))
        rec.update(_scalar_items(info))
        history.append(rec)
        if verbose:
            print(f"[{it:4d}] value={rec['value']:.6g} |g|={rec['grad_norm']:.3g}")
        stop = callback is not None and callback(it, x, rec)
        if checkpoint:
            _save_checkpoint(checkpoint, it + 1, x, state, history)
        if stop:
            break
    return OptResult(x=x, history=history)


# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #

class ModeTransmission:
    """Mode amplitude / power at a port plane, usable inside a JAX loss.

    ``port_monitor`` is a plane DFT monitor (normal x) recording
    ``('Ey','Ez','Hy','Hz')``; ``mode_result`` a :class:`ModeResult` solved on
    the same transverse grid. ``amplitude(out)`` is the complex modal
    amplitude per frequency (linear in the fields), ``obj(out)`` its power
    ``|alpha|**2``. With a unit-power one-way source this is the power
    transmission ``|S|**2`` (normalize by a reference run otherwise).
    ``direction`` selects the forward (``'+'``) or backward (``'-'``) wave.
    """

    def __init__(self, port_monitor: str, mode_result, mode_index: int = 0,
                 dA: float = 1.0, direction: str = "+"):
        from .smatrix import mode_port_fields
        if direction not in ("+", "-"):
            raise ValueError("direction must be '+' or '-'")
        self.port = port_monitor
        self.mode = mode_port_fields(mode_result, mode_index)
        self.dA = dA
        self.direction = direction

    def amplitude(self, out):
        from .smatrix import mode_amplitudes, _TANGENTIAL_X
        d = out["dft"][self.port]
        ap, am = mode_amplitudes({c: d[c] for c in _TANGENTIAL_X},
                                 self.mode, self.dA)
        return ap if self.direction == "+" else am

    def __call__(self, out):
        import jax.numpy as jnp
        return jnp.abs(self.amplitude(out)) ** 2


class FieldProjection:
    """Generic linear port readout: ``sum(weights * field)`` per frequency.

    A lightweight alternative to :class:`ModeTransmission` (e.g. on a 2-D
    grid): ``weights`` (broadcastable to one frequency slice of the DFT
    monitor's component) define the "mode" to project onto. ``amplitude`` is
    linear in the fields; ``obj(out)`` returns ``|amplitude|**2``.
    """

    def __init__(self, monitor: str, component: str, weights):
        self.monitor = monitor
        self.component = component
        self.weights = np.asarray(weights)

    def amplitude(self, out):
        f = out["dft"][self.monitor][self.component]
        w = self.weights
        return (f * w.conj()).reshape(f.shape[0], -1).sum(-1)

    def __call__(self, out):
        import jax.numpy as jnp
        return jnp.abs(self.amplitude(out)) ** 2


def combine(objs: Sequence[Callable], weights: Optional[Sequence[float]] = None,
            reduce: str = "mean"):
    """Merge several objectives into one scalar ``f(out)``.

    Each ``obj(out)`` may return a per-frequency array; it is averaged over
    frequency first. ``reduce='mean'`` takes the weighted mean across
    objectives, ``'min'`` the worst one (for a maximized figure of merit).
    """
    import jax.numpy as jnp
    w = np.ones(len(objs)) if weights is None else np.asarray(weights, float)
    if reduce not in ("mean", "min"):
        raise ValueError("reduce must be 'mean' or 'min'")

    def f(out):
        vals = jnp.stack([jnp.mean(o(out)) for o in objs])
        if reduce == "min":
            return jnp.min(vals)
        return jnp.sum(w * vals) / w.sum()
    return f
