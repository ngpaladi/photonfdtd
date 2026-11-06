"""Train a chip like a neural network.

The etch pattern (a :class:`~photonfdtd.DesignRegion` density) or the shape
parameters (:class:`~photonfdtd.ShapeDesign`) are the *weights*. A sample
``x`` is the vector of complex amplitudes driving the chip's ``N`` input
ports (optionally one vector per wavelength channel), and the model output is
what detectors read at its ``M`` output ports: the powers ``|a_out|**2``
(the network's nonlinearity), or the complex amplitudes themselves. The
weights are then fitted to a dataset ``(X, Y)`` by minibatch gradient
descent, with a classification or regression loss, like any other model.

Why this is cheap: Maxwell's equations are linear, so for fixed weights the
chip *is* a complex transfer matrix ``T`` of shape ``(n_freq, M, N)``. One
simulation per input port measures it exactly, and every sample of every
minibatch is then ``a_out = T @ x`` - a matrix product, not an FDTD run.
Differentiating the minibatch loss pulls ``dL/dT`` back through those same
``N`` simulations, so **an optimizer step costs N forward + N adjoint
simulations, independent of batch and dataset size**.

Coherent inputs are phasors at each wavelength: a complex ``x`` means
"drive port ``n`` with amplitude ``|x_n|`` and phase ``arg x_n``", which the
transfer matrix represents exactly even though each port simulation uses a
real time-domain pulse.
"""
from __future__ import annotations
from typing import Callable, Optional, Sequence

import numpy as np

from .optimize import OptResult, minimize


def _ce_of(design, sim):
    if hasattr(design, "ce_builder"):
        return design.ce_builder(sim)
    if callable(design):
        return design
    raise TypeError("design must have ce_builder(sim) or be ce_of(params)")


class PhotonicModel:
    """A chip as a differentiable model: ``params -> transfer matrix``.

    Parameters
    ----------
    sims : sequence of Simulation
        One simulation per input port, identical except for their sources
        (input ``n`` excited alone in ``sims[n]``): same grid, structures and
        monitors.
    design : DesignRegion | ShapeDesign | callable
        The trainable part; anything with ``ce_builder(sim)``, or a
        ``ce_of(params, **aux)`` coefficient builder.
    readouts : sequence
        One per output port, each with ``amplitude(out) -> (n_freq,)``
        complex, linear in the fields: :class:`~photonfdtd.ModeTransmission`
        or :class:`~photonfdtd.FieldProjection`.
    detection : {'power', 'coherent'}
        ``'power'`` returns ``sum_f |a|**2`` per output port, shape (B, M);
        ``'coherent'`` the complex amplitudes, shape (B, n_freq, M).
    """

    def __init__(self, sims: Sequence, design, readouts: Sequence,
                 detection: str = "power", remat: str = "nested",
                 checkpoint_levels: int = 2):
        import jax
        from .jaxbackend import _prepare
        if detection not in ("power", "coherent"):
            raise ValueError("detection must be 'power' or 'coherent'")
        self.sims = list(sims)
        if not self.sims:
            raise ValueError("need at least one input-port simulation")
        s0 = self.sims[0]
        for s in self.sims[1:]:
            if (tuple(s.grid.shape) != tuple(s0.grid.shape) or s.dt != s0.dt
                    or s.n_steps != s0.n_steps
                    or not np.array_equal(s.eps_r, s0.eps_r)):
                raise ValueError("input-port simulations must share grid, "
                                 "time step, run time and structures")
        self.readouts = list(readouts)
        self.detection = detection
        self.remat, self.levels = remat, checkpoint_levels
        self.ce_of = _ce_of(design, s0)
        self.preps = [_prepare(s) for s in self.sims]
        self._T_jit = jax.jit(self._transfer)

    @property
    def n_inputs(self) -> int:
        return len(self.sims)

    @property
    def n_outputs(self) -> int:
        return len(self.readouts)

    def _transfer(self, params, aux):
        import jax.numpy as jnp
        from .jaxbackend import _simulate_out
        ce = self.ce_of(params, **aux)
        cols = []
        for sim, prep in zip(self.sims, self.preps):
            out = _simulate_out(sim, prep, ce, self.remat, self.levels)
            cols.append(jnp.stack([jnp.atleast_1d(r.amplitude(out))
                                   for r in self.readouts], axis=-1))
        return jnp.stack(cols, axis=-1)                    # (F, M, N)

    def transfer(self, params, **aux):
        """The transfer matrix ``T`` (n_freq, M, N) for weights ``params``."""
        return self._T_jit(params, aux)

    def apply(self, T, X):
        """Model output for inputs ``X`` given a transfer matrix ``T``.

        ``X`` is (B, N) (the same amplitudes at every frequency) or
        (B, n_freq, N).
        """
        import jax.numpy as jnp
        X = jnp.asarray(X)
        if X.ndim == 2:
            X = jnp.broadcast_to(X[:, None, :], (X.shape[0], T.shape[0], X.shape[1]))
        a = jnp.einsum("fmn,bfn->bfm", T, X.astype(T.dtype))
        if self.detection == "power":
            return jnp.sum(jnp.abs(a) ** 2, axis=1)
        return a

    def predict(self, params, X, **aux):
        """Model output for ``X``: N simulations, then a matrix product."""
        return self.apply(self.transfer(params, **aux), X)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

def _cross_entropy(pred, y):
    """Classification on detected power: the class score is the fraction of
    the output power in that port, ``p_m = P_m / sum P``."""
    import jax.numpy as jnp
    p = pred / (jnp.sum(pred, axis=-1, keepdims=True) + 1e-300)
    py = jnp.take_along_axis(p, y[:, None].astype(jnp.int32), axis=-1)[:, 0]
    return -jnp.mean(jnp.log(py + 1e-12))


def _mse(pred, y):
    import jax.numpy as jnp
    return jnp.mean(jnp.abs(pred - y) ** 2)


_LOSSES = {"cross_entropy": _cross_entropy, "mse": _mse}


def accuracy(pred, y) -> float:
    """Fraction of samples whose brightest output port is the label."""
    return float(np.mean(np.argmax(np.asarray(pred), -1) == np.asarray(y)))


def train(model: PhotonicModel, params0, X, Y, loss="cross_entropy",
          optimizer=None, epochs: int = 10, batch_size: Optional[int] = None,
          schedule: Optional[Callable] = None, bounds=(0.0, 1.0), seed: int = 0,
          validation=None, callback: Optional[Callable] = None,
          checkpoint: Optional[str] = None, verbose: bool = False) -> OptResult:
    """Fit the chip's weights to ``(X, Y)`` by minibatch gradient descent.

    Parameters
    ----------
    model : PhotonicModel
    params0 : array
        Initial weights (e.g. a uniform 0.5 density).
    X : array
        Inputs, (n, N) or (n, n_freq, N), real or complex.
    Y : array
        Integer class labels (n,) for ``loss='cross_entropy'`` (class ``m``
        is output port ``m``); target outputs for ``'mse'``.
    loss : 'cross_entropy' | 'mse' | callable
        ``loss(pred, y_batch) -> scalar`` for a custom loss.
    optimizer : optax-style transformation, optional
        Defaults to ``Adam(0.01)``.
    epochs, batch_size : int
        ``batch_size=None`` takes full-batch steps (one per epoch). An
        incomplete last batch is dropped so the step compiles once.
    schedule : callable, optional
        ``schedule(step) -> dict`` of design hyper-parameters (e.g.
        :func:`~photonfdtd.beta_continuation`), counted in optimizer steps.
    seed : int
        Seeds the per-epoch shuffle (deterministic, resumable).
    validation : (X_val, Y_val), optional
        Evaluated at the end of every epoch (N extra forward simulations) and
        logged as ``val_loss`` / ``val_accuracy``.
    callback, checkpoint, verbose
        As for :func:`~photonfdtd.minimize`.

    Returns
    -------
    OptResult
        ``history`` holds one record per step with ``epoch``, ``value``
        (batch loss) and, for classification, the batch ``accuracy``.
    """
    import jax
    import jax.numpy as jnp

    lossf = _LOSSES[loss] if isinstance(loss, str) else loss
    classify = loss == "cross_entropy"
    Xj = jnp.asarray(X)
    Yj = jnp.asarray(Y)
    n = int(Xj.shape[0])
    bs = n if batch_size is None else int(batch_size)
    if not 0 < bs <= n:
        raise ValueError("batch_size must be in [1, len(X)]")
    steps_per_epoch = n // bs

    def batch_loss(params, idx, aux):
        pred = model.apply(model._transfer(params, aux), Xj[idx])
        y = Yj[idx]
        acc = (jnp.mean(jnp.argmax(pred, -1) == y) if classify
               else jnp.zeros([]))
        return lossf(pred, y), acc

    fn = jax.jit(jax.value_and_grad(batch_loss, has_aux=True))

    def vg(x, batch, epoch, **aux):
        (val, acc), g = fn(x, batch, aux)
        return val, g, ({"accuracy": acc} if classify else {})

    def sched(it):
        epoch, s = divmod(it, steps_per_epoch)
        perm = np.random.default_rng([seed, epoch]).permutation(n)
        aux = dict(schedule(it)) if schedule is not None else {}
        aux.update(batch=jnp.asarray(perm[s * bs:(s + 1) * bs]), epoch=epoch)
        return aux

    def cb(it, x, rec):
        if validation is not None and (it + 1) % steps_per_epoch == 0:
            aux = dict(schedule(it)) if schedule is not None else {}
            Xv, Yv = validation
            pred = model.predict(x, jnp.asarray(Xv), **aux)
            rec["val_loss"] = float(lossf(pred, jnp.asarray(Yv)))
            if classify:
                rec["val_accuracy"] = accuracy(pred, Yv)
        return callback(it, x, rec) if callback is not None else False

    return minimize(vg, params0, optimizer, n_iter=epochs * steps_per_epoch,
                    bounds=bounds, schedule=sched, callback=cb,
                    checkpoint=checkpoint, verbose=verbose)


def fit_transfer(model: PhotonicModel, params0, target_T, global_phase=True,
                 **kwargs) -> OptResult:
    """Train the chip to implement a given linear map ``target_T`` (M, N) or
    (n_freq, M, N), e.g. a splitter, a mode router or a DFT.

    The loss is the relative error ``sum |T - target_T|**2 / sum |target_T|**2``;
    with ``global_phase=True`` the
    (physically arbitrary) common phase of ``T`` is factored out first,
    ``||T||**2 + ||t||**2 - 2|<T, t>|``. Each step costs N forward + N adjoint
    simulations. ``kwargs`` go to :func:`~photonfdtd.minimize`.
    """
    import jax
    import jax.numpy as jnp

    t = jnp.asarray(target_T)
    if not np.any(np.asarray(target_T)):
        raise ValueError("target_T must not be all zero")

    def loss_T(params, aux):
        T = model._transfer(params, aux)
        tt = jnp.broadcast_to(t, T.shape).astype(T.dtype)
        if global_phase:
            err = (jnp.sum(jnp.abs(T) ** 2) + jnp.sum(jnp.abs(tt) ** 2)
                   - 2 * jnp.abs(jnp.sum(jnp.conj(T) * tt)))
        else:
            err = jnp.sum(jnp.abs(T - tt) ** 2)
        # Relative error, so the loss is O(1) whatever units T is in (Adam's
        # eps would otherwise swamp the gradient of a tiny raw loss).
        return err / jnp.sum(jnp.abs(tt) ** 2)

    fn = jax.jit(jax.value_and_grad(loss_T))

    def vg(x, **aux):
        return fn(x, aux)
    return minimize(vg, params0, **kwargs)
