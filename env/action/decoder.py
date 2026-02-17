from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

from ..specs import PARAM_DOMAIN_REGISTRY
from ..types import ParamDomain, ParamSpec
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
    Decode normalized design variables into physical parameter values.

    This component maps an RL-friendly normalized vector `x` to circuit
    parameter values (e.g., device widths, inductances, capacitances)
    according to a circuit-specific parameter domain registry.

    Conceptually:

        x in [0, max_param]^n   -->   physical parameters in [p_min, p_max]^n

    For each parameter spec `s`, decoding uses:
    - scaling: linear or log10 interpolation
    - optional significant-figure rounding
    - optional step snapping (quantization)
    - optional clipping of x to the valid normalized range

    Parameters
    ----------
    circuit_type : str, default="CS"
        Key into `PARAM_DOMAIN_REGISTRY` to select the parameter domain
        (set of parameter specs) for the target circuit.
    max_param : float, default=1.0
        Maximum value of the normalized parameter coordinate system.
        Typically 1.0 if `x` is already normalized to [0, 1], but can be
        larger if the upstream policy uses a different scale.
    clip_x : bool, default=True
        If True, clip input `x` into [0, max_param] before decoding.
        If False, values outside may raise errors downstream depending on
        registry/spec validation logic.

    Attributes
    ----------
    domain : ParamDomain
        Resolved parameter domain containing ordered parameter keys and
        parameter specs for `circuit_type`.

    Raises
    ------
    KeyError
        If `circuit_type` is not found in `PARAM_DOMAIN_REGISTRY`.
    ValueError
        If a parameter spec is ill-formed (e.g., unknown scale).
    ValueError
        If configuration values are invalid (via internal validators).

    Notes
    -----
    - `x` is treated as a *flat vector* with length equal to the number of
      parameters in the selected domain.
    - This class is frozen (immutable). The resolved `domain` is set once in
      `__post_init__`.
    """

    circuit_type: str = "CS"
    max_param: float = 1.0
    clip_x: bool = True

    domain: ParamDomain = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate configuration and resolve the circuit parameter domain.

        This method:
        1) Validates numeric config (e.g., max_param > 0 and finite)
        2) Loads the domain from `PARAM_DOMAIN_REGISTRY`
        3) Validates each ParamSpec within the domain (bounds, positivity, step)

        Raises
        ------
        KeyError
            If `circuit_type` is unknown.
        """
        _require_finite_positive("max_param", float(self.max_param))

        try:
            domain = PARAM_DOMAIN_REGISTRY[self.circuit_type]
        except KeyError as e:
            raise KeyError(
                f"Unknown circuit_type={self.circuit_type!r}. "
                f"Available: {tuple(PARAM_DOMAIN_REGISTRY.keys())}"
            ) from e

        object.__setattr__(self, "domain", domain)

        # Context string used by validators for error messages
        ctx = getattr(self.domain, "circuit_type", self.circuit_type)

        # Validate spec contracts: ordered bounds, log-scale positivity, step positivity
        for spec in self.domain.specs:
            _require_ordered_bounds(ctx, low=float(spec.p_min), high=float(spec.p_max))
            _require_log_positive_bounds(ctx, spec)
            _require_step_positive(ctx, spec)

    def _decode_one(self, x_i: float, spec: ParamSpec) -> float:
        """
        Decode a single normalized coordinate into a physical value.

        Parameters
        ----------
        x_i : float
            One element from the normalized design vector `x`.
            Interpreted in [0, max_param] then normalized to t in [0, 1].
        spec : ParamSpec
            Parameter specification describing scaling and constraints.

        Returns
        -------
        float
            Physical parameter value after interpolation and optional
            rounding/step snapping.

        Raises
        ------
        ValueError
            If `spec.scale` is not recognized.
        """
        # Normalize into [0, 1] and clip for safety (even if upstream clipped)
        t = _clip01_scalar(float(x_i) / float(self.max_param))

        # Interpolate based on parameter scale
        if spec.scale == "linear":
            v = _interp_linear(t, p_min=float(spec.p_min), p_max=float(spec.p_max))
        elif spec.scale == "log":
            v = _interp_log10(t, p_min=float(spec.p_min), p_max=float(spec.p_max))
        else:
            raise ValueError(
                f"Unknown scale={spec.scale!r} for parameter {spec.name!r}"
            )

        # Optional: round to significant figures (for numeric stability / reproducibility)
        if spec.round_sig_k is not None:
            v = float(_round_sig_scalar(v, int(spec.round_sig_k)))

        # Optional: snap to discrete step (quantization)
        if spec.step is not None:
            v = _snap_step_scalar(
                v,
                step=float(spec.step),
                p_min=float(spec.p_min),
                p_max=float(spec.p_max),
            )

        return float(v)

    def decode_x_to_physical(self, x: Sequence[float]) -> np.ndarray:
        """
        Decode a normalized vector `x` into physical parameter values.

        Parameters
        ----------
        x : Sequence[float]
            Normalized design vector. Must have length equal to the number
            of parameters in the selected domain.

        Returns
        -------
        np.ndarray
            Physical parameter vector of shape (n,) in float64.

        Raises
        ------
        ValueError
            If `x` does not have the expected length (via `_require_len`).

        Notes
        -----
        If `clip_x=True`, `x` is clipped to [0, max_param] before decoding.
        """
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)

        n = len(self.domain.keys)
        _require_len("x", x_arr, n)

        if self.clip_x:
            x_arr = np.clip(x_arr, 0.0, float(self.max_param))

        out = np.empty(n, dtype=np.float64)
        for i, spec in enumerate(self.domain.specs):
            out[i] = self._decode_one(float(x_arr[i]), spec)

        return out

    def make_design_config(self, x: Sequence[float]) -> Dict[str, float]:
        """
        Decode `x` and return a dict suitable for netlist/template rendering.

        Parameters
        ----------
        x : Sequence[float]
            Normalized design vector.

        Returns
        -------
        Dict[str, float]
            Mapping from parameter name to decoded physical value.
        """
        phys = self.decode_x_to_physical(x)
        keys = self.domain.keys.as_tuple()
        return {k: float(v) for k, v in zip(keys, phys)}

    def variable_keys(self) -> Tuple[str, ...]:
        """
        Return the ordered parameter keys for the current domain.

        Returns
        -------
        Tuple[str, ...]
            Parameter names in the same order expected by `decode_x_to_physical`.
        """
        return self.domain.keys.as_tuple()
