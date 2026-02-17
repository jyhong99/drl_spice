from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from ngspice.errors import CircuitNetlistNotFound, NetlistFormatError, NetlistMarkerError
from ngspice.patterns import (
    _DESIGNVAR_END,
    _DESIGNVAR_START,
    _DEVICE_END,
    _RE_ASSIGN_BARE,
    _RE_ASSIGN_BRACED,
    _RE_BRACED_VAR,
    _RE_LEAD_INST,
    _RE_PARAM,
    _RE_SUBCKT,
)
from ngspice.types import CircuitNames, PathLike


# =============================================================================
# Circuit
# =============================================================================

class Circuit:
    """
    Parse a rendered ngspice netlist and expose design-variable/device mappings.

    This class loads a netlist file and builds three maps:

    - `dsgnvar_to_val`:
        Design variable name -> numeric value parsed from the design-variable block.
        (Typically from `.param <name> = <number>` lines.)
    - `dvc_to_dsgnvar`:
        Device instance attribute -> design variable name, inferred from the device block.
        Keys may be either:
          * "<inst>_<key>" for key=value-style assignments (e.g., "M1_W"),
          * or "<inst>" for fallback mappings when no key can be inferred.
    - `dvc_to_val`:
        Device mapping key -> numeric value (looked up through `dsgnvar_to_val`),
        or None if the variable is missing.

    The netlist is assumed to contain sentinel markers (see `ngspice.patterns`):
    - `** design_variables_start` ... `** design_variables_end`
    - `**** begin user architecture code` (end boundary for device parsing)
    and a `.subckt` line that defines the start of the circuit section.

    Parameters
    ----------
    netlist_path:
        Path to the rendered netlist file on disk.
    names:
        `CircuitNames` containing:
        - `designvar_names`: ordered tuple of optimizable variable names
        - `device_names`: ordered tuple of instance names to consider (M1, M2, R1, ...)

    Attributes
    ----------
    netlist_path:
        Netlist file path as string.
    netlist:
        Full netlist text loaded from disk (decoded with `errors="replace"`).
    designvar_names:
        Design variable names (list form; order preserved).
    device_names:
        Device instance names (list form; order preserved).
    dsgnvar_to_val:
        Mapping from design variable name to numeric value.
    dvc_to_dsgnvar:
        Mapping from device attribute key to design variable name.
    dvc_to_val:
        Mapping from device attribute key to numeric value (or None).

    Notes
    -----
    - Name matching for devices and variables is case-insensitive by using
      lower-cased sets (`_dev_set`, `_var_set`) for membership tests.
    - This class is intentionally strict about marker structure to avoid
      silently parsing the wrong region of the netlist.
    """

    def __init__(self, netlist_path: PathLike, *, names: CircuitNames) -> None:
        self.netlist_path = str(netlist_path)

        # Keep canonical ordered lists for downstream reproducibility/logging.
        self.designvar_names = list(names.designvar_names)
        self.device_names = list(names.device_names)

        # Membership sets for case-insensitive matching.
        self._dev_set = {d.lower() for d in self.device_names}
        self._var_set = {v.lower() for v in self.designvar_names}

        self.netlist = self._load_netlist()
        self.dsgnvar_to_val, self.dvc_to_dsgnvar, self.dvc_to_val = self._update_circuit()

    # -------------------------------------------------------------------------
    # Netlist loading / marker helpers
    # -------------------------------------------------------------------------

    def _load_netlist(self) -> str:
        """
        Load netlist text from `self.netlist_path`.

        Returns
        -------
        str
            Full netlist text.

        Raises
        ------
        CircuitNetlistNotFound
            If the netlist file does not exist.

        Notes
        -----
        - Uses `errors="replace"` to avoid crashing on encoding issues.
        """
        p = Path(self.netlist_path)
        if not p.exists():
            raise CircuitNetlistNotFound(p)
        return p.read_text(errors="replace")

    @staticmethod
    def _find_between_markers(text: str, start: str, end: str) -> Tuple[int, int]:
        """
        Find a substring region bounded by start/end marker strings.

        Parameters
        ----------
        text:
            Full netlist text to search.
        start:
            Start marker string. The returned region begins immediately after it.
        end:
            End marker string. The returned region ends immediately before it.

        Returns
        -------
        (start_idx, end_idx) : tuple[int, int]
            Indices such that `text[start_idx:end_idx]` is the content between markers.

        Raises
        ------
        NetlistMarkerError
            If start marker is missing, or end marker does not occur after start.

        Notes
        -----
        - This is a strict positional search using `str.find`.
        - End marker must appear *after* the start marker; otherwise we treat the
          netlist as malformed.
        """
        s = text.find(start)
        if s < 0:
            raise NetlistMarkerError(f"Start marker not found: {start!r}")

        e = text.find(end, s + len(start))
        if e < 0:
            raise NetlistMarkerError(f"End marker not found after start: end={end!r}")

        return s + len(start), e

    def _find_designvar_block(self) -> Tuple[int, int]:
        """
        Locate the design-variable `.param` block region.

        Returns
        -------
        (start_idx, end_idx) : tuple[int, int]
            Region bounded by `_DESIGNVAR_START` and `_DESIGNVAR_END`.
        """
        return self._find_between_markers(self.netlist, _DESIGNVAR_START, _DESIGNVAR_END)

    def _find_device_block(self) -> Tuple[int, int]:
        """
        Locate the device instantiation region to scan for variable references.

        The device block is defined as:
        - start: the `.subckt` line (or its comment-wrapped variant)
        - end: `_DEVICE_END` marker ("begin user architecture code")

        Returns
        -------
        (start_idx, end_idx) : tuple[int, int]
            Region within which instance lines are analyzed.

        Raises
        ------
        NetlistMarkerError
            If the device-end marker is missing.
        NetlistFormatError
            If `.subckt` start cannot be located, or marker ordering is invalid.
        """
        e = self.netlist.find(_DEVICE_END)
        if e < 0:
            raise NetlistMarkerError(f"Device end marker not found: {_DEVICE_END!r}")

        m = _RE_SUBCKT.search(self.netlist)
        if not m:
            raise NetlistFormatError("Could not locate subckt start in netlist.")
        s = m.start()

        if s >= e:
            raise NetlistFormatError(
                "subckt start is after user-architecture marker (malformed netlist)."
            )
        return s, e

    # -------------------------------------------------------------------------
    # Mapping builders
    # -------------------------------------------------------------------------

    def _map_dsgnvar_to_val(self) -> Dict[str, float]:
        """
        Parse the design-variable block and build a {designvar -> float} mapping.

        Returns
        -------
        dict[str, float]
            Mapping from variable name to numeric value.

        Raises
        ------
        NetlistMarkerError
            If the design-variable marker block cannot be located.

        Notes
        -----
        - Uses `_RE_PARAM` which matches numeric-literal RHS only.
          If your `.param` includes expressions, use a different parser.
        """
        s, e = self._find_designvar_block()
        block = self.netlist[s:e]

        out: Dict[str, float] = {}
        for m in _RE_PARAM.finditer(block):
            out[m.group(1)] = float(m.group(2))
        return out

    def _map_device_to_dsgnvar(self) -> Dict[str, str]:
        """
        Infer mappings from device instance attributes to design variables.

        For each instance line inside the device block, if the instance name is
        recognized (in `self.device_names`), we attempt to associate parameters
        on that line with known design variables using four strategies:

        1) key={var}
           Example: "M1 ... W={W_m1}"  -> maps "M1_W" -> "W_m1"
        2) key=var
           Example: "M1 ... W=W_m1"    -> maps "M1_W" -> "W_m1"
        3) fallback: first {var} occurrence (no explicit key)
           Example: "M1 ... {W_m1}"    -> maps "M1"   -> "W_m1"
        4) fallback: bare token match
           Example: "M1 ... W_m1"      -> maps "M1"   -> "W_m1"

        Returns
        -------
        dict[str, str]
            Mapping from a device-attribute key to a design variable name.

        Raises
        ------
        NetlistMarkerError
            If the device boundary marker is missing.
        NetlistFormatError
            If `.subckt` cannot be located or boundary ordering is invalid.

        Notes
        -----
        - Matching of device names and variables is case-insensitive.
        - This function is intentionally conservative: it only maps to variables
          that are listed in `names.designvar_names`.
        """
        s, e = self._find_device_block()
        block = self.netlist[s:e]

        dvc_to_dsgnvar: Dict[str, str] = {}

        for raw_line in block.splitlines():
            line = raw_line.strip()

            # Skip empty lines, comments, and continuation lines.
            if not line or line.startswith("*") or line.startswith("+"):
                continue

            # Identify leading token as a candidate instance name.
            m_inst = _RE_LEAD_INST.match(line)
            if not m_inst:
                continue

            inst = m_inst.group(1)
            inst_l = inst.lower()
            if inst_l not in self._dev_set:
                continue

            toks = re.split(r"\s+", line)

            # Track whether this instance line produced any mapping.
            found_any = False

            # 1) key={var}
            for key, var in _RE_ASSIGN_BRACED.findall(line):
                if var.lower() in self._var_set:
                    dvc_to_dsgnvar[f"{inst}_{key}"] = var
                    found_any = True

            # 2) key=var
            for key, var in _RE_ASSIGN_BARE.findall(line):
                if var.lower() in self._var_set:
                    dvc_to_dsgnvar[f"{inst}_{key}"] = var
                    found_any = True

            # 3) fallback: first {var} occurrence
            if not found_any:
                for var in _RE_BRACED_VAR.findall(line):
                    if var.lower() in self._var_set:
                        dvc_to_dsgnvar[inst] = var
                        found_any = True
                        break

            # 4) fallback: bare token match
            if not found_any:
                for tok in toks:
                    if tok.lower() in self._var_set:
                        dvc_to_dsgnvar[inst] = tok
                        break

        return dvc_to_dsgnvar

    def _update_circuit(self) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, Optional[float]]]:
        """
        Build all derived maps from the currently loaded netlist.

        Returns
        -------
        (dsgnvar_to_val, dvc_to_dsgnvar, dvc_to_val)
            dsgnvar_to_val:
                {designvar -> float}
            dvc_to_dsgnvar:
                {device_attr_key -> designvar}
            dvc_to_val:
                {device_attr_key -> float or None}
        """
        dsgnvar_to_val = self._map_dsgnvar_to_val()
        dvc_to_dsgnvar = self._map_device_to_dsgnvar()
        dvc_to_val: Dict[str, Optional[float]] = {
            dvc: dsgnvar_to_val.get(var) for dvc, var in dvc_to_dsgnvar.items()
        }
        return dsgnvar_to_val, dvc_to_dsgnvar, dvc_to_val

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def refresh(self) -> None:
        """
        Recompute mappings using the currently loaded `self.netlist`.

        Notes
        -----
        - Does not re-read the file from disk. Use `reload()` for that.
        """
        self.dsgnvar_to_val, self.dvc_to_dsgnvar, self.dvc_to_val = self._update_circuit()

    def reload(self) -> None:
        """
        Reload netlist text from disk and recompute mappings.

        Raises
        ------
        CircuitNetlistNotFound
            If the netlist file no longer exists.
        """
        self.netlist = self._load_netlist()
        self.refresh()

    def get_designvar_to_val(self) -> Dict[str, float]:
        """
        Convenience accessor for parsing the design-variable block.

        Returns
        -------
        dict[str, float]
            Mapping from design variable name to numeric value.
        """
        return self._map_dsgnvar_to_val()
