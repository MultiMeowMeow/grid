"""Centralized feature selection for OPF inputs.

These constants define which columns participate in model inputs (and therefore
in normalization) for every structure.  Any column omitted here is ignored by
both the `OPFNormalizer` and `OPFCore`, so a single change updates the entire
pipeline.
"""

from __future__ import annotations

from typing import Sequence

# Bus -----------------------------------------------------------------------
# bus.x layout: [base_kv, bus_type, vmin, vmax]
BUS_TYPE_COL: int = 1
BUS_NUMERIC_INPUT_COLS: Sequence[int] = (0,)  # Only base_kv is used

# AC line -------------------------------------------------------------------
# edge_attr layout: [angmin, angmax, b_fr, b_to, r, x, rate_a, rate_b, rate_c]
AC_LINE_INPUT_COLS: Sequence[int] = (2, 3, 4, 5, 6, 7, 8)

# Transformer ---------------------------------------------------------------
# edge_attr layout:
#   [angmin, angmax, r, x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to]
TRANSFORMER_INPUT_COLS: Sequence[int] = (2, 3, 4, 5, 6, 7)

# Convenience lookups -------------------------------------------------------
EDGE_INPUT_COLS = {
    ("bus", "ac_line", "bus"): AC_LINE_INPUT_COLS,
    ("bus", "transformer", "bus"): TRANSFORMER_INPUT_COLS,
}

