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

# Feature columns that should be normalized per structure.  Everything omitted
# here is intentionally ignored (e.g. graph.baseMVA, bus voltage limits,
# ac_line angle bounds, transformer angle/phase/tap susceptances, loads, etc.).
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

        The network inputs we normalize (bus, generator, shunt and branch
        attributes) are constant across samples, so aggregating across a
        dataset is unnecessary.
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

        if "bus" in sample.node_types:
            _stats(sample["bus"].x, ("bus", "x"))
        if "generator" in sample.node_types:
            _stats(sample["generator"].x, ("generator", "x"))
        if "shunt" in sample.node_types:
            _stats(sample["shunt"].x, ("shunt", "x"))

        if ("bus", "ac_line", "bus") in sample.edge_types:
            _stats(
                sample[("bus", "ac_line", "bus")].edge_attr,
                (("bus", "ac_line", "bus"), "edge_attr"),
            )
        if ("bus", "transformer", "bus") in sample.edge_types:
            _stats(
                sample[("bus", "transformer", "bus")].edge_attr,
                (("bus", "transformer", "bus"), "edge_attr"),
            )

        return self

    @torch.no_grad()
    def normalize(self, data: HeteroData) -> HeteroData:
        def _normalize_subset(x: Tensor, key: PathKey, cols: Sequence[int]) -> Tensor:
            mean, std = self.stats_.get(key, (None, None))
            if mean is None:
                return x
            x = x.clone()
            idx = torch.as_tensor(cols, device=x.device)
            x[:, idx] = (x[:, idx] - mean[idx]) / std[idx]
            return x

        if "bus" in data.node_types:
            data["bus"].x = _normalize_subset(data["bus"].x, ("bus", "x"), BUS_NORMALIZED_COLS)

        if "generator" in data.node_types:
            mean, std = self.stats_.get(("generator", "x"), (None, None))
            if mean is not None:
                data["generator"].x = (data["generator"].x - mean) / std

        if "shunt" in data.node_types:
            mean, std = self.stats_.get(("shunt", "x"), (None, None))
            if mean is not None:
                data["shunt"].x = (data["shunt"].x - mean) / std

        if ("bus", "ac_line", "bus") in data.edge_types:
            key = (("bus", "ac_line", "bus"), "edge_attr")
            data[("bus", "ac_line", "bus")].edge_attr = _normalize_subset(
                data[("bus", "ac_line", "bus")].edge_attr, key, AC_LINE_NORMALIZED_COLS
            )

        if ("bus", "transformer", "bus") in data.edge_types:
            key = (("bus", "transformer", "bus"), "edge_attr")
            data[("bus", "transformer", "bus")].edge_attr = _normalize_subset(
                data[("bus", "transformer", "bus")].edge_attr, key, TRANSFORMER_NORMALIZED_COLS
            )

        return data

    def denormalize(self, x: Tensor, key: PathKey, cols=None) -> Tensor:
        mean, std = self.stats_.get(key, (None, None))
        if mean is None:
            return x
        if cols is None:
            return x * std + mean
        x = x.clone()
        x[:, cols] = x[:, cols] * std[cols] + mean[cols]
        return x

    def save(self, path: str):
        torch.save(self, path)

    @classmethod
    def load(cls, path: str) -> OPFNormalizer:
        return torch.load(path, map_location="cpu")