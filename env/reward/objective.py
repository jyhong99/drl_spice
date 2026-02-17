from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..specs import get_perf_spec, perf_metric_order
from ..utils.common_utils import (
    _require_finite_positive,
    _require_len,
)
from ..utils.reward_utils import (
    _as_1d_f64,
    _require_finite_nonneg,
    _require_spec_ref_separation,
    _safe_denom,
    _u_vec,
    _is_non_convergent,
)
from ..patterns import (
    F_KEY,
    METRICS_KEY,
    NONCONV_KEY,
    REF_KEY,
    SAT_KEY,
    SPEC_KEY,
    U_KEY,
    UNSTABLE_KEY,
    VIOL_KEY,
    VIOL_TERM_KEY,
    FEASIBLE_KEY,
    K_MIN_KEY,
    MIN_OBJ_LEN,
    PD_FLOOR,
    NF_DB_MIN,
    NF_DB_MAX,
    DEFAULT_BAD_FOM,
    ENABLE_LIN_KEY,
    IIP3_INDEX,
    INVALID_KEY,
    PERF_KEYS_KEY,
)


@dataclass
class ConstraintModel:
    """
    Constraint evaluation model for multi-metric circuit performance.

    This model converts raw performance metrics `o` into a normalized constraint
    satisfaction vector `f` and a violation vector `viol`, based on a per-metric
    (spec, reference) pair.

    Definitions
    ----------
    Let, for each metric i:

    - spec_i : target specification value (threshold)
    - ref_i  : reference baseline value (normalization anchor)
    - o_i    : observed metric value

    A normalized satisfaction score is computed as:

        denom_i = safe_denom(spec_i, ref_i)
        f_i = (o_i - ref_i) / denom_i

    Then the non-negative violation is:

        viol_i = max(0, 1 - f_i)

    So:
    - `viol_i = 0` implies the constraint is satisfied (within tolerance).
    - Larger `viol_i` indicates stronger violation.

    Aggregation
    -----------
    A weighted p-norm-like scalar violation term is computed:

        viol_term = ( (sum_i w_i * viol_i^p) / (sum_i w_i + eps_wsum) )^(1/p)

    This scalar is useful as a single constraint penalty term.

    Additional invalidity signals
    -----------------------------
    The model can also flag invalid states using the auxiliary `u` vector:

    - non-convergent simulation
    - unstable design (e.g., K-factor below threshold)

    Parameters
    ----------
    circuit_type : str, default="CS"
        Circuit type key used to fetch the performance specification object.
    enable_linearity : bool, default=False
        If True, include linearity-related metrics (e.g., IIP3) in the metric list.
        This also affects `perf_metric_order(...)` and the spec registry.
    p : float, default=2.0
        Exponent used in the violation aggregation (p-norm style).
    w : np.ndarray or None, default=None
        Metric weights. If None, uses ones for all metrics.
        Must have shape (n_metrics,).
    use_k : bool, default=True
        If True, consider `K_MIN_KEY` (if present) as an instability indicator.
    k_min : float, default=1.0
        Threshold for instability when `use_k` is True.
    eps_denom : float, default=1e-12
        Small value used in safe denominator computation to prevent divide-by-zero.
    eps_viol0 : float, default=1e-12
        Tolerance for treating a violation as "zero" (satisfied).
    eps_wsum : float, default=1e-12
        Small value added to weight sum to avoid division by zero.

    Attributes
    ----------
    perf_keys : tuple[str, ...]
        Ordered metric keys used to interpret the performances vector.
    perf_spec : Any
        Performance spec object loaded from the spec registry. Must provide:
        - `.as_dict()` -> per-metric spec dict containing `"value"`
        - `.references` -> dict mapping metric key to reference value

    Raises
    ------
    ValueError
        If configuration is invalid (weights shape mismatch, spec missing keys, etc.).

    Notes
    -----
    - `performances` must align with `perf_keys` order exactly.
    - The `metrics` dict returned merges the performance vector with any `aux` fields.
    """

    circuit_type: str = "CS"
    enable_linearity: bool = False

    p: float = 2.0
    w: Optional[np.ndarray] = None

    use_k: bool = True
    k_min: float = 1.0

    eps_denom: float = 1e-12
    eps_viol0: float = 1e-12
    eps_wsum: float = 1e-12

    perf_keys: tuple[str, ...] = field(init=False, repr=False)
    perf_spec: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate configuration and load performance specification.

        This method:
        1) Validates numeric hyperparameters.
        2) Resolves the ordered metric keys (perf_keys).
        3) Loads the circuit performance spec object.
        4) Initializes and validates weights.
        5) Validates that every perf_key has both a spec value and reference value.
        6) Validates spec/ref separation to avoid degenerate normalization.

        Raises
        ------
        ValueError
            If weight shapes mismatch, or spec entries are missing/invalid.
        """
        _require_finite_positive("p", float(self.p))
        _require_finite_nonneg("eps_denom", float(self.eps_denom))
        _require_finite_nonneg("eps_viol0", float(self.eps_viol0))
        _require_finite_nonneg("eps_wsum", float(self.eps_wsum))
        _require_finite_positive("k_min", float(self.k_min))

        self.perf_keys = tuple(perf_metric_order(enable_linearity=bool(self.enable_linearity)))
        self.perf_spec = get_perf_spec(
            circuit_type=str(self.circuit_type),
            enable_linearity=bool(self.enable_linearity),
        )

        n = len(self.perf_keys)
        if self.w is None:
            self.w = np.ones(n, dtype=np.float64)
        else:
            w = _as_1d_f64(self.w)
            if w.shape != (n,):
                raise ValueError(f"w must have shape {(n,)}, got {w.shape}")
            self.w = w

        spec_dict = self.perf_spec.as_dict()
        ctx = getattr(self.perf_spec, "circuit_type", self.circuit_type)

        # Ensure spec + reference exist for each metric key
        for k in self.perf_keys:
            if k not in spec_dict:
                raise ValueError(f"[{ctx}] perf_spec missing item: {k}")
            if k not in self.perf_spec.references:
                raise ValueError(f"[{ctx}] perf_spec missing reference: {k}")

        # Validate spec/ref separation to avoid denom ~ 0 normalization
        for k in self.perf_keys:
            spec_v = float(spec_dict[k]["value"])
            ref_v = float(self.perf_spec.references[k])
            _require_spec_ref_separation(
                ctx=str(ctx),
                key=str(k),
                spec_v=spec_v,
                ref_v=ref_v,
                eps_denom=float(self.eps_denom),
            )

    def _to_metrics_dict(
        self,
        performances: np.ndarray,
        aux: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Convert vector-form performances into a named metrics dictionary.

        Parameters
        ----------
        performances : np.ndarray
            Performance vector aligned to `self.perf_keys`.
        aux : dict[str, float] or None
            Optional auxiliary metrics to merge into the metrics dict.

        Returns
        -------
        dict[str, float]
            Mapping from metric key to numeric value.

        Raises
        ------
        ValueError
            If `performances` does not have the expected length.
        """
        o = _as_1d_f64(performances)
        _require_len("performances", o, len(self.perf_keys))

        d: Dict[str, float] = {k: float(v) for k, v in zip(self.perf_keys, o)}
        for k, v in (aux or {}).items():
            d[str(k)] = float(v)
        return d

    def compute(
        self,
        performances: np.ndarray,
        aux: Optional[Dict[str, float]] = None,
        *,
        non_convergent: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Compute constraint satisfaction, violations, and feasibility signals.

        Parameters
        ----------
        performances : np.ndarray
            Performance vector aligned with `self.perf_keys`.
        aux : dict[str, float] or None, default=None
            Extra metrics (e.g., K-factor minima, bandwidth, etc.).
            These are merged into the returned `METRICS_KEY` dict.
        non_convergent : bool or None, default=None
            If provided, overrides automatic non-convergence detection.
            If None, uses `_is_non_convergent(o)` on the performance vector.

        Returns
        -------
        dict[str, Any]
            A structured dictionary containing:

            - F_KEY : np.ndarray
                Normalized satisfaction vector `f`.
            - VIOL_KEY : np.ndarray
                Per-metric violation vector `viol = max(0, 1-f)`.
            - VIOL_TERM_KEY : float
                Weighted p-norm aggregate violation.
            - SAT_KEY : bool
                True if all violations are <= eps_viol0.
            - FEASIBLE_KEY : bool
                True if satisfied and not invalid (non-convergent/unstable).
            - U_KEY : np.ndarray
                Auxiliary invalidity indicator vector (e.g., [nonconv, unstable]).
            - NONCONV_KEY : bool
                Non-convergence flag extracted from `u`.
            - UNSTABLE_KEY : bool
                Instability flag extracted from `u`.
            - INVALID_KEY : bool
                True if any invalidity flag is active.
            - METRICS_KEY : dict[str, float]
                Named metrics dict (performances + aux merged).
            - SPEC_KEY : np.ndarray
                Spec vector aligned with perf_keys.
            - REF_KEY : np.ndarray
                Reference vector aligned with perf_keys.
            - PERF_KEYS_KEY : tuple[str, ...]
                Metric key order used by all vectors.
            - ENABLE_LIN_KEY : bool
                Whether linearity metrics were enabled.

        Notes
        -----
        - The satisfaction scaling uses both spec and reference values; this
          tends to be more numerically stable than using spec alone when metrics
          have very different magnitudes.
        - Feasibility requires both (i) satisfied performance constraints and
          (ii) not being flagged invalid by `u`.
        """
        metrics = self._to_metrics_dict(performances, aux)

        spec_d = self.perf_spec.as_dict()
        spec = np.array([float(spec_d[k]["value"]) for k in self.perf_keys], dtype=np.float64)
        ref = np.array([float(self.perf_spec.references[k]) for k in self.perf_keys], dtype=np.float64)
        o = np.array([float(metrics[k]) for k in self.perf_keys], dtype=np.float64)

        if non_convergent is None:
            non_convergent = _is_non_convergent(o)

        denom = _safe_denom(spec, ref, eps_denom=float(self.eps_denom))
        f = (o - ref) / denom
        viol = np.maximum(0.0, 1.0 - f)

        w = _as_1d_f64(self.w)
        p = float(self.p)
        wsum = float(np.sum(w)) + float(self.eps_wsum)
        viol_term = float((np.dot(w, viol**p) / wsum) ** (1.0 / p))

        satisfied_perf = bool(np.all(viol <= float(self.eps_viol0)))

        unstable = False
        if self.use_k and (K_MIN_KEY in metrics):
            unstable = bool(float(metrics[K_MIN_KEY]) < float(self.k_min))

        u = _u_vec(non_convergent=bool(non_convergent), unstable=bool(unstable))
        invalid = bool((u[0] == 1.0) or (u[1] == 1.0))
        feasible = bool((not invalid) and satisfied_perf)

        return {
            F_KEY: f,
            VIOL_KEY: viol,
            VIOL_TERM_KEY: viol_term,
            SAT_KEY: satisfied_perf,
            FEASIBLE_KEY: feasible,
            U_KEY: u,
            NONCONV_KEY: bool(u[0] == 1.0),
            UNSTABLE_KEY: bool(u[1] == 1.0),
            INVALID_KEY: invalid,
            METRICS_KEY: metrics,
            SPEC_KEY: spec,
            REF_KEY: ref,
            PERF_KEYS_KEY: self.perf_keys,
            ENABLE_LIN_KEY: bool(self.enable_linearity),
        }


@dataclass
class ObjectiveModel:
    """
    Objective model for computing FoM-based scalar objectives.

    This class computes a (heuristic) figure-of-merit (FoM) in dB from a
    performance vector. The objective for optimization is typically the
    *negative* FoM (i.e., minimize J = -FoM).

    Parameters
    ----------
    eps : float, default=1e-9
        Small constant used to avoid log(0) and division-by-zero.

    Notes
    -----
    Expected performance vector layout for base FoM:
    ``[S11_dB, S21_dB, S22_dB, NF_dB, PD_mW, ...]``

    If linearity is enabled and `IIP3_INDEX` is valid, IIP3 may be added
    to the FoM as an additional bonus term (if finite).
    """

    eps: float = 1e-9

    def fom_db(self, performances: np.ndarray) -> float:
        """
        Compute figure-of-merit (FoM) in dB from a performance vector.

        Parameters
        ----------
        performances : np.ndarray
            Performance vector. Must contain at least `MIN_OBJ_LEN` entries
            in the expected order:
            ``[S11_dB, S21_dB, S22_dB, NF_dB, PD_mW]``.

            If `performances` also contains an entry at `IIP3_INDEX` and it is
            finite, the FoM is incremented by `IIP3_dBm`.

        Returns
        -------
        float
            FoM value in dB. Returns `DEFAULT_BAD_FOM` if the input is invalid.

        Notes
        -----
        The FoM is computed using a common RF-LNA style combination:

        - Convert return losses in dB to linear magnitudes
        - Convert NF(dB) to noise factor
        - Penalize NF and power consumption
        - Reward gain (S21) and matching (1 - |S11|, 1 - |S22|)

        This is a heuristic score; its exact form should match your paper/benchmark
        definition.
        """
        o = _as_1d_f64(performances)
        if o.shape[0] < MIN_OBJ_LEN:
            return float(DEFAULT_BAD_FOM)

        s11_db, s21_db, s22_db, nf_db, pd_mw = [float(x) for x in o[:MIN_OBJ_LEN]]

        if not np.isfinite([s11_db, s21_db, s22_db, nf_db, pd_mw]).all():
            return float(DEFAULT_BAD_FOM)

        # Practical clipping for stability
        pd_mw = max(pd_mw, PD_FLOOR)
        nf_db = min(max(nf_db, NF_DB_MIN), NF_DB_MAX)

        # Convert dB metrics to linear quantities
        s11 = 10.0 ** (s11_db / 20.0)
        s22 = 10.0 ** (s22_db / 20.0)
        nf = 10.0 ** (nf_db / 10.0)

        one_minus_in = max(1.0 - s11, float(self.eps))
        one_minus_out = max(1.0 - s22, float(self.eps))
        noisefactor = max(nf - 1.0, 0.0)

        fom = float(
            20.0 * np.log10(one_minus_in)
            + s21_db
            + 20.0 * np.log10(one_minus_out)
            - 10.0 * np.log10(noisefactor + float(self.eps))
            - 10.0 * np.log10(float(pd_mw) + float(self.eps))
        )

        # Optional: add linearity bonus (e.g., IIP3)
        if o.shape[0] >= (IIP3_INDEX + 1):
            iip3_dbm = float(o[IIP3_INDEX])
            if np.isfinite(iip3_dbm):
                fom += iip3_dbm

        return float(fom)

    def j_fom(self, performances: np.ndarray) -> float:
        """
        Compute optimization objective value from FoM.

        Parameters
        ----------
        performances : np.ndarray
            Performance vector.

        Returns
        -------
        float
            Objective scalar. By default: ``J = -FoM``.
        """
        return -float(self.fom_db(performances))

    @staticmethod
    def var_from_f(f: np.ndarray) -> float:
        """
        Compute variance of a satisfaction vector.

        Parameters
        ----------
        f : np.ndarray
            Satisfaction vector.

        Returns
        -------
        float
            Variance of `f` (float).
        """
        return float(np.var(_as_1d_f64(f)))
