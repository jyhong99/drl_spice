from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

from ..specs import PARAM_DOMAIN_REGISTRY
from ..types import ParamDomain
from ..utils.common_utils import (
    _require_len,
    _require_ordered_bounds,
    _require_finite_positive,
)
from ..utils.action_utils import (
    _require_log_positive_bounds,
    _require_step_positive,
    _clip01_scalar,
    _snap_step_scalar,
    _interp_linear,
    _interp_log10,
    _round_sig_scalar,
)


@dataclass(frozen=True)
class SpecDecoder:
    """
    Decode normalized policy coordinates into physical circuit parameters.

    ``SpecDecoder`` maps a normalized vector ``x`` to physical design values
    according to a circuit-specific parameter domain.

    For each parameter ``i`` with bounds ``[p_min_i, p_max_i]`` and normalized
    coordinate ``t_i = clip(x_i / max_param, 0, 1)``:

    - linear scale:

      .. math::

          v_i = p_{\min,i} + t_i (p_{\max,i} - p_{\min,i})

    - log scale:

      .. math::

          v_i = 10^{\log_{10}(p_{\min,i}) + t_i (\log_{10}(p_{\max,i}) - \log_{10}(p_{\min,i}))}

    Optional post-processing per parameter:

    - significant-digit rounding
    - step-size snapping (quantization)

    Parameters
    ----------
    circuit_type : str, default="CS"
        Circuit key used to resolve ``PARAM_DOMAIN_REGISTRY``.
    max_param : float, default=1.0
        Upper bound of the normalized coordinate system.
    clip_x : bool, default=True
        If True, input ``x`` is clipped to ``[0, max_param]`` before decode.
        If False, out-of-range values raise ``ValueError``.

    Attributes
    ----------
    domain : ParamDomain
        Resolved domain for ``circuit_type``.

    Notes
    -----
    - The decode path is vectorized for interpolation to reduce Python loop
      overhead in frequent step-time calls.
    - Rounding and step snapping are applied only where configured.
    """

    circuit_type: str = "CS"
    max_param: float = 1.0
    clip_x: bool = True

    domain: ParamDomain = field(init=False, repr=False)

    # Cached decode structures for efficiency.
    _max_param: float = field(init=False, repr=False)
    _keys: Tuple[str, ...] = field(init=False, repr=False)
    _n: int = field(init=False, repr=False)
    _p_min: np.ndarray = field(init=False, repr=False)
    _p_max: np.ndarray = field(init=False, repr=False)
    _p_span: np.ndarray = field(init=False, repr=False)
    _is_log: np.ndarray = field(init=False, repr=False)
    _log_min: np.ndarray = field(init=False, repr=False)
    _log_span: np.ndarray = field(init=False, repr=False)
    _has_log: bool = field(init=False, repr=False)
    _round_cfg: Tuple[Tuple[int, int], ...] = field(init=False, repr=False)
    _step_cfg: Tuple[Tuple[int, float], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate configuration, resolve domain, and precompute decode caches.

        Raises
        ------
        KeyError
            If ``circuit_type`` is not present in ``PARAM_DOMAIN_REGISTRY``.
        ValueError
            If bounds or per-parameter decode specs are invalid.
        """
        max_param = _require_finite_positive("max_param", float(self.max_param))

        ct = str(self.circuit_type).strip().upper()
        if not ct:
            raise ValueError("circuit_type must be a non-empty string")

        try:
            domain = PARAM_DOMAIN_REGISTRY[ct]
        except KeyError as e:
            raise KeyError(
                f"Unknown circuit_type={self.circuit_type!r}. "
                f"Available: {tuple(PARAM_DOMAIN_REGISTRY.keys())}"
            ) from e

        object.__setattr__(self, "circuit_type", ct)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "_max_param", max_param)

        ctx = getattr(domain, "circuit_type", ct)
        specs = domain.specs
        n = len(domain.keys)

        p_min = np.empty(n, dtype=np.float64)
        p_max = np.empty(n, dtype=np.float64)
        is_log = np.zeros(n, dtype=bool)
        round_cfg: list[Tuple[int, int]] = []
        step_cfg: list[Tuple[int, float]] = []

        for i, spec in enumerate(specs):
            pmin_i = float(spec.p_min)
            pmax_i = float(spec.p_max)

            _require_ordered_bounds(ctx, low=pmin_i, high=pmax_i)
            _require_log_positive_bounds(ctx, spec)
            _require_step_positive(ctx, spec)

            p_min[i] = pmin_i
            p_max[i] = pmax_i
            is_log[i] = (spec.scale == "log")

            if spec.scale not in ("linear", "log"):
                raise ValueError(f"Unknown scale={spec.scale!r} for parameter {spec.name!r}")

            if spec.round_sig_k is not None:
                round_cfg.append((i, int(spec.round_sig_k)))
            if spec.step is not None:
                step_cfg.append((i, float(spec.step)))

        log_min = np.zeros(n, dtype=np.float64)
        log_span = np.zeros(n, dtype=np.float64)
        has_log = bool(np.any(is_log))
        if has_log:
            log_min[is_log] = np.log10(p_min[is_log])
            log_span[is_log] = np.log10(p_max[is_log]) - log_min[is_log]

        object.__setattr__(self, "_keys", domain.keys.as_tuple())
        object.__setattr__(self, "_n", n)
        object.__setattr__(self, "_p_min", p_min)
        object.__setattr__(self, "_p_max", p_max)
        object.__setattr__(self, "_p_span", p_max - p_min)
        object.__setattr__(self, "_is_log", is_log)
        object.__setattr__(self, "_log_min", log_min)
        object.__setattr__(self, "_log_span", log_span)
        object.__setattr__(self, "_has_log", has_log)
        object.__setattr__(self, "_round_cfg", tuple(round_cfg))
        object.__setattr__(self, "_step_cfg", tuple(step_cfg))

    def _decode_one(self, x_i: float, i: int) -> float:
        """
        Decode one coordinate by parameter index.

        Parameters
        ----------
        x_i : float
            Single normalized coordinate.
        i : int
            Parameter index into the resolved domain.

        Returns
        -------
        float
            Decoded physical value after interpolation.

        Notes
        -----
        This helper is used for debugging and parity checks. Main decode is
        vectorized in :meth:`decode_x_to_physical`.
        """
        t = _clip01_scalar(float(x_i) / self._max_param)
        if self._is_log[i]:
            return _interp_log10(t, p_min=float(self._p_min[i]), p_max=float(self._p_max[i]))
        return _interp_linear(t, p_min=float(self._p_min[i]), p_max=float(self._p_max[i]))

    def decode_x_to_physical(self, x: Sequence[float]) -> np.ndarray:
        """
        Decode a normalized vector into physical parameter values.

        Parameters
        ----------
        x : array_like of float
            Normalized input vector. After flattening, shape must be ``(n,)``
            where ``n = len(variable_keys())``.

        Returns
        -------
        numpy.ndarray
            Physical parameter vector with shape ``(n,)`` and dtype ``float64``.

        Raises
        ------
        ValueError
            If input shape is incorrect, contains non-finite values, or violates
            the configured normalized range when ``clip_x=False``.

        Notes
        -----
        Execution steps:

        1. Flatten and validate ``x``.
        2. Clip or validate range in normalized coordinates.
        3. Vectorized interpolation for all dimensions.
        4. Optional per-dimension significant-digit rounding.
        5. Optional per-dimension step snapping.
        """
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)

        _require_len("x", x_arr, self._n)
        if not np.all(np.isfinite(x_arr)):
            raise ValueError("x must contain only finite values")

        if self.clip_x:
            x_norm = np.clip(x_arr, 0.0, self._max_param)
        else:
            x_min = float(np.min(x_arr))
            x_max = float(np.max(x_arr))
            if x_min < 0.0 or x_max > self._max_param:
                raise ValueError(
                    f"x out of bounds [0, {self._max_param}] with clip_x=False: min={x_min}, max={x_max}"
                )
            x_norm = x_arr

        t = np.clip(x_norm / self._max_param, 0.0, 1.0)

        out = self._p_min + t * self._p_span
        if self._has_log:
            out_log = np.power(
                10.0,
                self._log_min[self._is_log] + t[self._is_log] * self._log_span[self._is_log],
            )
            out[self._is_log] = out_log

        # Optional per-dimension post-processing.
        for i, k in self._round_cfg:
            out[i] = _round_sig_scalar(float(out[i]), k)

        for i, step in self._step_cfg:
            out[i] = _snap_step_scalar(
                float(out[i]),
                step=step,
                p_min=float(self._p_min[i]),
                p_max=float(self._p_max[i]),
            )

        return out

    def make_design_config(self, x: Sequence[float]) -> Dict[str, float]:
        """
        Decode ``x`` and package values as a netlist-friendly mapping.

        Parameters
        ----------
        x : array_like of float
            Normalized vector accepted by :meth:`decode_x_to_physical`.

        Returns
        -------
        dict[str, float]
            Ordered mapping from parameter name to physical value.

        Examples
        --------
        >>> dec = SpecDecoder(circuit_type="CS")
        >>> cfg = dec.make_design_config([0.5] * len(dec.variable_keys()))
        >>> isinstance(cfg, dict)
        True
        """
        phys = self.decode_x_to_physical(x)
        return {k: float(v) for k, v in zip(self._keys, phys)}

    def variable_keys(self) -> Tuple[str, ...]:
        """
        Return ordered design-variable names for this decoder.

        Returns
        -------
        tuple[str, ...]
            Variable keys in the exact order expected by
            :meth:`decode_x_to_physical`.
        """
        return self._keys
