from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Union, Sequence

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from feature_layout import EDGE_INPUT_COLS, NODE_INPUT_COLS

PathKey = Tuple[Union[str, Tuple[str, str, str]], str]  # e.g. ("bus","x") or (("bus","ac_line","bus"),"edge_attr")
Stats = Tuple[Tensor, Tensor]
EPS = 1e-8


NODE_NORMALIZED_COLS = NODE_INPUT_COLS
EDGE_NORMALIZED_COLS = EDGE_INPUT_COLS


@dataclass
class OPFNormalizer:
    """
    Standardize continuous input features of an OPF sample.  The following
    inputs are normalized (column selection defined in ``feature_layout``):
      - bus.x numeric inputs
      - generator.x inputs
      - load.x inputs
      - shunt.x inputs
      - physical edge attributes (AC lines & transformers)

    Everything else is intentionally untouched, including graph-level baseMVA
    and all supervised targets.
    """
    stats_: Dict[PathKey, Stats] = field(default_factory=dict)


    def fit(self, sample: HeteroData) -> OPFNormalizer:
        """Estimate statistics from a single grid sample.
        """

        def _stats(x: Tensor, key: PathKey):
            if x is None or x.numel() == 0:
                return
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            std = torch.sqrt(torch.clamp(var, min=0.0))
            std = torch.where(std < EPS, torch.full_like(std, EPS), std)
            self.stats_[key] = (mean, std)

        self.stats_.clear()

        for ntype, cols in NODE_NORMALIZED_COLS.items():
            if len(cols) == 0 or ntype not in sample.node_types:
                continue
            _stats(sample[ntype].x[:, cols], (ntype, "x"))

        for etype, cols in EDGE_NORMALIZED_COLS.items():
            if len(cols) == 0 or etype not in sample.edge_types:
                continue
            _stats(sample[etype].edge_attr[:, cols], (etype, "edge_attr"))

        return self

    @torch.no_grad()
    def normalize(self, data: HeteroData) -> HeteroData:
        def _normalize_subset(x: Tensor, key: PathKey, cols: Sequence[int]) -> Tensor:
            if len(cols) == 0:
                return x
            stats = self.stats_.get(key)
            if stats is None:
                return x
            mean, std = stats
            mean = mean.to(x.device)
            std = std.to(x.device)
            idx = torch.as_tensor(cols, device=x.device)
            x_sel = x[:, idx]
            x[:, idx] = (x_sel - mean) / std
            return x

        for ntype, cols in NODE_NORMALIZED_COLS.items():
            if ntype not in data.node_types:
                continue
            data[ntype].x = _normalize_subset(data[ntype].x, (ntype, "x"), cols)

        for etype, cols in EDGE_NORMALIZED_COLS.items():
            key = (etype, "edge_attr")
            if etype not in data.edge_types:
                continue
            data[etype].edge_attr = _normalize_subset(data[etype].edge_attr, key, cols)
        return data

    def save(self, path: str):
        torch.save(self, path)

    @classmethod
    def load(cls, path: str) -> OPFNormalizer:
        return torch.load(path, map_location="cuda")