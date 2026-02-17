from __future__ import annotations

import re

"""
Regular-expression primitives for ngspice text/RAW-ascii parsing and template
pre/post-processing.

This module centralizes compiled regex patterns and sentinel marker strings so
that:
- parsing logic is consistent across readers,
- patterns are compiled once (performance),
- schema/format assumptions are explicit and testable.

Notes
-----
- Many patterns use MULTILINE (re.M) and DOTALL (re.S) to operate on blocks
  spanning multiple lines.
- Case-insensitive matching is enabled where ngspice output can vary in casing.
- Marker strings (e.g., "** design_variables_start") are used to locate
  user-defined blocks inside template netlists.
"""


# =============================================================================
# Numeric token grammar
# =============================================================================

_NUM_PAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
"""
Regex fragment for a floating-point literal.

This is a *fragment* (not compiled) intended to be embedded in other patterns.

Matches
-------
- Integers: "0", "12", "-7"
- Decimals: "1.0", "1.", ".5", "-.25"
- Scientific notation: "1e3", "-2.5E-6"

Notes
-----
- This does not match NaN/Inf tokens; those should be handled separately if
  needed.
- This fragment is designed for ASCII numeric tokens as typically produced by
  ngspice and hand-written netlists.
"""


_RE_HEADER_INT_TPL = r"{key}\s*:\s*(\d+)"
"""
Template regex for integer-valued header fields.

Parameters
----------
key:
    Placeholder to be substituted by caller (e.g., "No. Variables", "No. Points").

Returns
-------
str
    A regex string that captures an integer value as group(1).

Notes
-----
- This is not compiled because the `key` is usually dynamic and substituted
  before compilation.
"""


# =============================================================================
# RAW-ascii parsing (Variables/Values blocks)
# =============================================================================

_RE_VARS_BLOCK = re.compile(
    r"^\s*Variables\s*:?\s*\r?\n(.*?)^\s*Values\s*:?\s*\r?\n",
    flags=re.I | re.M | re.S,
)
"""
Match the 'Variables' block up to (but not including) the 'Values' header.

Captures
--------
group(1):
    The entire text between the Variables header line and the Values header line.

Use case
--------
- Extract variable definitions (index/name/type) before parsing numeric samples.

Notes
-----
- Uses DOTALL so the captured block may span multiple lines.
- Uses MULTILINE so ^ matches the beginning of each line in the text.
- Allows optional ':' after header keywords and optional CRLF newlines.
"""


_RE_VALUES_HDR_EOL = re.compile(
    r"^\s*Values\s*:?\s*\r?\n",
    flags=re.I | re.M,
)
"""
Match the 'Values' header line *including the trailing newline*.

Use case
--------
- Used when slicing raw text:
    values_text = text[m.end():]
- Ensures the returned slice starts exactly at the first data line.
- Intentionally consumes the end-of-line to avoid manual offset handling.

Notes
-----
- This is a *text-splitting* helper, not a logical section validator.
"""


_RE_VALUES_HDR = re.compile(
    r"(?im)^\s*Values\s*:?\s*$"
)
"""
Match a single 'Values' header line (line-level sentinel).

Use case
--------
- Detect the presence/location of the 'Values' section header.
- Suitable for line-by-line parsing or structural validation.

Notes
-----
- Does NOT consume the trailing newline.
- Should NOT be used for text slicing without additional handling.
"""


_RE_SPLIT_WS = re.compile(r"[\t ]+")
"""
Split pattern for whitespace-separated tokens.

Use case
--------
- Parsing NGSPICE RAW ASCII tables where columns are space/tab separated.
- Used after line-by-line processing.

Notes
-----
- Does NOT match commas.
- Newlines are intentionally excluded.
"""


_RE_SPLIT = re.compile(r"[,\t ]+")
"""
Split pattern for comma- or whitespace-separated tokens.

Use case
--------
- More permissive parsing when:
    - commas may appear (CSV-like output),
    - ngspice `wrdata` formatting varies by version/settings.

Notes
-----
- Useful for defensive parsing.
- Slightly less strict than `_RE_SPLIT_WS`.
"""


_RE_INT_PREFIX = re.compile(r"^\d+\b")
"""
Match an integer prefix at the beginning of a line.

Use case
--------
- Many ngspice value lines begin with a point index (e.g., "0  1.23  4.56").
  This pattern detects the leading index for optional stripping.
"""


_RE_INT_TOKEN = re.compile(r"^\d+$")
"""
Match a whole-string integer token.

Use case
--------
- Validate that a token is purely an integer index (no sign, no decimals).
"""


_RE_NUM_TOKEN = re.compile(_NUM_PAT)
"""
Match a numeric token (floating-point literal) using `_NUM_PAT`.

Notes
-----
- This is not anchored; it can match a numeric substring inside a larger string.
  Anchor with ^...$ if you require full-token validation.
"""


# =============================================================================
# Template markers (sentinel strings)
# =============================================================================

_DESIGNVAR_START = "** design_variables_start"
_DESIGNVAR_END = "** design_variables_end"
_DEVICE_END = "**** begin user architecture code"
_ANALYSIS_KNOB_START = "** analysis_knobs_start"
_ANALYSIS_KNOB_END = "** analysis_knobs_end"
_LET_KNOB_START = "** let_knobs_start"
_LET_KNOB_END = "** let_knobs_end"

"""
Sentinel marker strings used to locate sections inside ngspice templates.

These markers are intended to be placed in netlist template files and then
searched for during rendering or post-processing.

Conventions
-----------
- Markers start with '*' so they are treated as comments by SPICE.
- Double-asterisk and quadruple-asterisk prefixes are used to reduce collision
  risk with ordinary comments.

Common blocks
-------------
- design_variables: parameters controlled by the optimizer/RL agent
- analysis_knobs: analysis-level metadata (z0, corner, sweep settings, etc.)
- let_knobs: `.let`-style computed expressions or derived parameters
- device_end: a boundary marker before user architecture code
"""


# =============================================================================
# Netlist parameter / structure patterns
# =============================================================================

_RE_PARAM = re.compile(rf"(?i)\.param\s+(\w+)\s*=\s*({_NUM_PAT})")
"""
Match a .param definition with a numeric literal value.

Captures
--------
group(1):
    Parameter name (\\w+).
group(2):
    Numeric literal value using `_NUM_PAT`.

Examples matched
---------------
- ".param R1=50"
- ".PARAM gm = 1e-3"

Notes
-----
- This pattern intentionally accepts only numeric literals on the RHS.
  If you allow expressions (e.g., ".param x = {y}*2"), use a different pattern.
"""


_RE_SUBCKT = re.compile(r"(?im)^[ \t]*\*{0,2}\.subckt\b")
"""
Detect a (possibly commented) `.subckt` line.

Use case
--------
- Some templates may have `.subckt` preceded by up to two comment asterisks to
  disable/enable blocks. This pattern still detects such lines.

Notes
-----
- MULTILINE is required so ^ anchors at each line.
- Case-insensitive because `.subckt` casing can vary.
"""


_RE_LEAD_INST = re.compile(r"^\s*([^\s*]+)")
"""
Capture the leading token of a netlist line as a potential instance name.

Captures
--------
group(1):
    First non-whitespace token that is not '*' (to avoid pure comment lines).

Use case
--------
- Extract instance identifiers like 'M1', 'Rload', 'XU1', etc.

Caveats
-------
- This does not validate SPICE instance naming rules; it only captures tokens.
"""


_RE_BRACED_VAR = re.compile(r"\{(\w+)\}")
"""
Match a brace-wrapped variable reference like '{foo}'.

Captures
--------
group(1):
    Variable name inside braces.

Use case
--------
- Jinja-like / brace substitution contracts in templates.
"""


_RE_ASSIGN_BRACED = re.compile(r"(?i)\b([A-Za-z_]\w*)\s*=\s*\{(\w+)\}")
"""
Match an assignment where RHS is a braced variable: 'lhs = {rhs}'.

Captures
--------
group(1):
    LHS identifier (parameter/attribute).
group(2):
    RHS variable name inside braces.

Examples matched
---------------
- "W = {W_m1}"
- "gain = {target_gain}"
"""


_RE_ASSIGN_BARE = re.compile(r"(?i)\b([A-Za-z_]\w*)\s*=\s*(\w+)\b")
"""
Match a simple bare assignment: 'lhs = rhs' (identifier-to-identifier).

Captures
--------
group(1):
    LHS identifier.
group(2):
    RHS identifier.

Examples matched
---------------
- "model = tt"
- "corner = ss"

Caveats
-------
- This intentionally does not accept numeric RHS; use `_RE_PARAM` for numeric
  `.param` lines.
- Because it is relatively permissive, apply it only in contexts where you
  expect simple assignments (e.g., controlled knob blocks), not arbitrary netlist
  lines.
"""
