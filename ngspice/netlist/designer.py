from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ngspice.errors import (
    DesignVarCountMismatch,
    DesignerError,
    NetlistPatchError,
    NetlistWriteError,
    TargetNetlistDesignVarCountMismatch,
)
from ngspice.netlist.circuit import Circuit
from ngspice.netlist.patcher import patch_block
from ngspice.patterns import (
    _ANALYSIS_KNOB_END,
    _ANALYSIS_KNOB_START,
    _DESIGNVAR_END,
    _DESIGNVAR_START,
    _LET_KNOB_END,
    _LET_KNOB_START,
)
from ngspice.textio import atomic_write
from ngspice.types import Marker


# =============================================================================
# Designer
# =============================================================================

class Designer:
    """
    Patch a circuit netlist in-place by injecting design variables and knobs.

    The `Designer` is responsible for modifying a target circuit netlist text
    by replacing content inside marker-delimited blocks. Typical use cases:

    - Update `.param` design variables (controlled by an optimizer/RL agent).
    - Inject optional "analysis knobs" (meta parameters controlling analyses).
    - Inject optional `.let`-style expressions ("let knobs").

    Markers (sentinel strings) are configured via `ngspice.patterns` and are
    typically present as SPICE comments in the template netlist.

    Parameters
    ----------
    num_design_variables:
        Optional expected number of design variables. When set, the designer can
        enforce that the provided config length matches and optionally validate
        the existing netlist's marker block contains the expected count.
    strict_count:
        If True, mismatched counts raise errors. If False, count mismatches are
        allowed (but may still be validated depending on your use).

    Attributes
    ----------
    design_marker:
        Marker delimiting the design-variable block.
    analysis_knobs_marker:
        Marker delimiting the analysis-knobs block.
    let_knobs_marker:
        Marker delimiting the let-knobs block.
    num_design_variables:
        Expected design-variable count (or None for no enforcement).
    strict_count:
        Whether to strictly enforce/validate counts.

    Notes
    -----
    - The patching mechanism is delegated to `patch_block`, which replaces the
      content between marker boundaries.
    - This class writes the patched netlist back to `target_circuit.netlist_path`
      using an atomic write pattern.
    """

    def __init__(self, *, num_design_variables: Optional[int] = None, strict_count: bool = True) -> None:
        if num_design_variables is not None and int(num_design_variables) < 0:
            raise ValueError("num_design_variables must be >= 0 or None")
        self.design_marker = Marker(_DESIGNVAR_START, _DESIGNVAR_END)
        self.analysis_knobs_marker = Marker(_ANALYSIS_KNOB_START, _ANALYSIS_KNOB_END)
        self.let_knobs_marker = Marker(_LET_KNOB_START, _LET_KNOB_END)

        self.num_design_variables = None if num_design_variables is None else int(num_design_variables)
        self.strict_count = bool(strict_count)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def design_circuit(
        self,
        target_circuit: Circuit,
        *,
        design_variables_config: Mapping[str, Any],
        analysis_knobs_config: Optional[Mapping[str, Any]] = None,
        let_knobs_config: Optional[Mapping[str, Any]] = None,
        sort_keys: bool = True,
        require_design_markers: bool = True,
    ) -> None:
        """
        Patch the target circuit netlist using the provided configuration maps.

        Parameters
        ----------
        target_circuit:
            Circuit instance whose `netlist` text and `netlist_path` will be
            updated in-place.
        design_variables_config:
            Mapping of design-variable names to values. Rendered as `.param`
            lines in the design-variable marker block.
        analysis_knobs_config:
            Optional mapping rendered as `.param` lines in the analysis-knobs
            marker block.
        let_knobs_config:
            Optional mapping rendered as `let` lines in the let-knobs marker block.
        sort_keys:
            If True, keys are sorted before rendering lines. This improves
            determinism and diff friendliness.
        require_design_markers:
            If True, design-variable markers must exist; otherwise patching the
            design-variable block raises.

        Raises
        ------
        DesignVarCountMismatch
            If `strict_count=True` and the provided design-variable config length
            mismatches `num_design_variables`.
        TargetNetlistDesignVarCountMismatch
            If `strict_count=True` and the currently parsed marker block inside
            the netlist does not contain the expected count.
        NetlistPatchError
            If patching any marker block fails.
        NetlistWriteError
            If writing the patched netlist back to disk fails.

        Notes
        -----
        - The netlist text is patched in memory first, then written to disk
          using `atomic_write`.
        - After writing, the circuit object is refreshed to update derived maps
          (design-var values, device mappings, etc.).
        """
        self._check_design_params(target_circuit, design_variables_config)
        self._check_keys_non_empty("design_variables_config", design_variables_config)
        if analysis_knobs_config is not None:
            self._check_keys_non_empty("analysis_knobs_config", analysis_knobs_config)
        if let_knobs_config is not None:
            self._check_keys_non_empty("let_knobs_config", let_knobs_config)

        text = target_circuit.netlist

        # --- Patch design variables (required or optional depending on flag)
        try:
            text = patch_block(
                text,
                marker=self.design_marker,
                params=design_variables_config,
                line_builder=self._param_line,
                sort_keys=sort_keys,
                required=require_design_markers,
            )
        except Exception as e:
            raise NetlistPatchError(
                f"Failed to patch design-variable block (required={require_design_markers})."
            ) from e

        # --- Patch analysis knobs (optional)
        if analysis_knobs_config is not None:
            try:
                text = patch_block(
                    text,
                    marker=self.analysis_knobs_marker,
                    params=analysis_knobs_config,
                    line_builder=self._param_line,
                    sort_keys=sort_keys,
                    required=False,
                )
            except Exception as e:
                raise NetlistPatchError("Failed to patch analysis-knobs block.") from e

        # --- Patch let knobs (optional)
        if let_knobs_config is not None:
            try:
                text = patch_block(
                    text,
                    marker=self.let_knobs_marker,
                    params=let_knobs_config,
                    line_builder=self._let_line,
                    sort_keys=sort_keys,
                    required=False,
                )
            except Exception as e:
                raise NetlistPatchError("Failed to patch let-knobs block.") from e

        # --- Write patched netlist to disk (atomic)
        try:
            atomic_write(Path(target_circuit.netlist_path), text)
        except Exception as e:
            # If your textio layer already raises TextWriteError, you may choose
            # to catch that specifically and attach richer context here.
            raise NetlistWriteError(f"Failed to write patched netlist: {target_circuit.netlist_path}") from e

        # --- Update in-memory circuit text and refresh derived mappings
        target_circuit.netlist = text
        self._refresh_circuit(target_circuit)

    # -------------------------------------------------------------------------
    # Validation helpers
    # -------------------------------------------------------------------------

    def _check_design_params(self, circuit: Circuit, params: Mapping[str, Any]) -> None:
        """
        Validate design-variable configuration against expected counts.

        Parameters
        ----------
        circuit:
            Target circuit used for best-effort validation of current netlist.
        params:
            Proposed design variable mapping.

        Raises
        ------
        DesignVarCountMismatch
            If strict count enforcement is enabled and config length mismatches
            the expected `num_design_variables`.
        DesignerError
            If parsing current netlist design variables fails under strict mode.
        TargetNetlistDesignVarCountMismatch
            If strict mode is enabled and the current netlist marker block does
            not contain the expected number of variables.

        Notes
        -----
        - If `num_design_variables` is None, no validation is performed.
        - "Best effort" validation reads the current design-variable block from
          the netlist so you detect stale/malformed templates early.
        """
        if self.num_design_variables is None:
            return

        expected = int(self.num_design_variables)
        got = len(params)

        if self.strict_count and got != expected:
            raise DesignVarCountMismatch(expected=expected, got=got)

        # Validate current netlist's design-var count inside markers (best effort)
        if self.strict_count:
            try:
                current = circuit._map_dsgnvar_to_val()  # parses marker-bounded `.param` region
            except Exception as e:
                raise DesignerError("Failed to parse design variables from target circuit netlist.") from e

            if len(current) != expected:
                raise TargetNetlistDesignVarCountMismatch(expected=expected, parsed=len(current))

    @staticmethod
    def _check_keys_non_empty(name: str, params: Mapping[str, Any]) -> None:
        """
        Validate that mapping keys are non-empty after string normalization.

        Parameters
        ----------
        name:
            Human-readable mapping name used in error messages.
        params:
            Mapping to validate.

        Raises
        ------
        DesignerError
            If any key becomes empty after ``str(key).strip()``.
        """
        for k in params:
            if not str(k).strip():
                raise DesignerError(f"{name} contains an empty key after normalization")

    # -------------------------------------------------------------------------
    # Line builders
    # -------------------------------------------------------------------------

    @staticmethod
    def _param_line(k: str, v: Any) -> str:
        """
        Render one `.param` line.

        Parameters
        ----------
        k:
            Parameter name.
        v:
            Value (typically numeric; may be string/expression depending on your template).

        Returns
        -------
        str
            Rendered `.param` line.
        """
        return f".param {k} = {v}"

    @staticmethod
    def _let_line(k: str, v: Any) -> str:
        """
        Render one `let` line.

        Parameters
        ----------
        k:
            Let variable name.
        v:
            Let expression/value.

        Returns
        -------
        str
            Rendered `let` line.
        """
        return f"let {k} = {v}"

    # -------------------------------------------------------------------------
    # Circuit refresh
    # -------------------------------------------------------------------------

    @staticmethod
    def _refresh_circuit(circuit: Circuit) -> None:
        """
        Refresh derived circuit mappings after netlist text changes.

        This method prefers a public `refresh()` API when available. If not,
        it falls back to a legacy internal method `_update_circuit()`.

        Parameters
        ----------
        circuit:
            Circuit instance to refresh.

        Notes
        -----
        - Prefer exposing `Circuit.refresh()` publicly (as you already do) so
          this method almost always takes the first path.
        """
        if hasattr(circuit, "refresh") and callable(getattr(circuit, "refresh")):
            circuit.refresh()
            return

        if hasattr(circuit, "_update_circuit") and callable(getattr(circuit, "_update_circuit")):
            circuit.dsgnvar_to_val, circuit.dvc_to_dsgnvar, circuit.dvc_to_val = circuit._update_circuit()
            return

        raise DesignerError("Target circuit does not provide refresh() or _update_circuit()")
