from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Union, Sequence

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from feature_layout import (
    BUS_NUMERIC_INPUT_COLS,
    AC_LINE_INPUT_COLS,
    TRANSFORMER_INPUT_COLS,
)

PathKey = Tuple[Union[str, Tuple[str, str, str]], str]  # e.g. ("bus","x") or (("bus","ac_line","bus"),"edge_attr")
Stats = Tuple[Tensor, Tensor]
EPS = 1e-8


BUS_NORMALIZED_COLS: Sequence[int] = BUS_NUMERIC_INPUT_COLS
AC_LINE_NORMALIZED_COLS: Sequence[int] = AC_LINE_INPUT_COLS
TRANSFORMER_NORMALIZED_COLS: Sequence[int] = TRANSFORMER_INPUT_COLS


@dataclass
class OPFNormalizer:
    """
    Standardize continuous input features of an OPF sample.  The following
    inputs are normalized:
      - bus.x: only base_kv (bus_type/vmin/vmax are skipped)
      - generator.x: all columns
      - shunt.x: all columns
      - (bus, ac_line, bus).edge_attr: all columns except angmin/angmax
      - (bus, transformer, bus).edge_attr: all columns except
        angmin/angmax/shift/b_fr/b_to

    Everything else is intentionally untouched, including graph-level baseMVA,
    loads (pd/qd already z-scored upstream) and all supervised targets.
    """
    stats_: Dict[PathKey, Stats] = field(default_factory=dict)


    def fit(self, sample: HeteroData) -> OPFNormalizer:
        """Estimate statistics from a single grid sample.
        """

        def _stats(x: Tensor, key: PathKey):
            if x is None:
                return
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            std = torch.sqrt(torch.clamp(var, min=0.0))
            std = torch.where(std < EPS, torch.full_like(std, EPS), std)
            self.stats_[key] = (mean, std)

        self.stats_.clear()

        _stats(sample["bus"].x, ("bus", "x"))
        _stats(sample["generator"].x, ("generator", "x"))
        _stats(sample["shunt"].x, ("shunt", "x"))
        _stats(
            sample[("bus", "ac_line", "bus")].edge_attr,
            (("bus", "ac_line", "bus"), "edge_attr"),
        )
        _stats(
            sample[("bus", "transformer", "bus")].edge_attr,
            (("bus", "transformer", "bus"), "edge_attr"),
        )

        return self

    @torch.no_grad()
    def normalize(self, data: HeteroData) -> HeteroData:
        def _normalize_subset(x: Tensor, key: PathKey, cols: Sequence[int]) -> Tensor:
            mean, std = self.stats_.get(key, (None, None))
            idx = torch.as_tensor(cols, device=x.device)
            x[:, idx] = (x[:, idx] - mean[idx]) / std[idx]
            return x

        data["bus"].x = _normalize_subset(data["bus"].x, ("bus", "x"), BUS_NORMALIZED_COLS)

        mean, std = self.stats_.get(("generator", "x"), (None, None))
        data["generator"].x = (data["generator"].x - mean) / std

        mean, std = self.stats_.get(("shunt", "x"), (None, None))
        data["shunt"].x = (data["shunt"].x - mean) / std

        key = (("bus", "ac_line", "bus"), "edge_attr")
        data[("bus", "ac_line", "bus")].edge_attr = _normalize_subset(
            data[("bus", "ac_line", "bus")].edge_attr, key, AC_LINE_NORMALIZED_COLS
        )
        key = (("bus", "transformer", "bus"), "edge_attr")
        data[("bus", "transformer", "bus")].edge_attr = _normalize_subset(
            data[("bus", "transformer", "bus")].edge_attr, key, TRANSFORMER_NORMALIZED_COLS
        )
        return data

    def save(self, path: str):
        torch.save(self, path)

    @classmethod
    def load(cls, path: str) -> OPFNormalizer:
        return torch.load(path, map_location="cuda")