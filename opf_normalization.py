from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.data import HeteroData, Dataset

PathKey = Tuple[Union[str, Tuple[str, str, str]], str]  # e.g. ("bus","x") or (("bus","ac_line","bus"),"edge_attr")
Stats = Tuple[Tensor, Tensor]
EPS = 1e-8


@dataclass
class OPFNormalizer:
    """
    Standardize continuous input features of OPFDataset:
      - bus.x: [base_kv, bus_type, vmin, vmax]  → normalize cols [0,2,3]
      - generator.x: all
      - load.x: all
      - shunt.x: all
      - (bus, ac_line, bus).edge_attr: all
      - (bus, transformer, bus).edge_attr: all

    Skip labels (bus.y, generator.y, edge_label, objective) and bus_type (discrete).
    """
    stats_: Dict[PathKey, Stats] = field(default_factory=dict)


    def fit(self, dataset: Dataset) -> OPFNormalizer:
        accum: Dict[PathKey, Tuple[Tensor, Tensor, int]] = {}

        def _accum(x: Tensor, key: PathKey):
            n = x.size(0)
            s, ss = x.sum(0), (x * x).sum(0)
            if key in accum:
                ps, pss, pn = accum[key]
                accum[key] = (ps + s, pss + ss, pn + n)
            else:
                accum[key] = (s, ss, n)

        for data in dataset:
            _accum(data["bus"].x, ("bus", "x"))
            _accum(data["generator"].x, ("generator", "x"))
            _accum(data["load"].x, ("load", "x"))
            _accum(data["shunt"].x, ("shunt", "x"))
            _accum(data[("bus", "ac_line", "bus")].edge_attr, (("bus", "ac_line", "bus"), "edge_attr"))
            _accum(data[("bus", "transformer", "bus")].edge_attr, (("bus", "transformer", "bus"), "edge_attr"))

        # finalize
        self.stats_.clear()
        for key, (s, ss, n) in accum.items():
            mean = s / n
            var = ss / n - mean ** 2
            std = torch.sqrt(torch.clamp(var, min=0.0))
            std = torch.where(std < EPS, torch.full_like(std, EPS), std)
            self.stats_[key] = (mean, std)
        return self

    @torch.no_grad()
    def normalize(self, data: HeteroData) -> HeteroData:
        # bus.x: skip col 1 (bus_type)
        mean, std = self.stats_[("bus", "x")]
        bus_x = data["bus"].x.clone()
        for c in [0, 2, 3]:
            bus_x[:, c] = (bus_x[:, c] - mean[c]) / std[c]
        data["bus"].x = bus_x

        # generator.x
        mean, std = self.stats_[("generator", "x")]
        data["generator"].x = (data["generator"].x - mean) / std
        # load.x
        mean, std = self.stats_[("load", "x")]
        data["load"].x = (data["load"].x - mean) / std
        # shunt.x
        mean, std = self.stats_[("shunt", "x")]
        data["shunt"].x = (data["shunt"].x - mean) / std
        # ac_line.edge_attr
        mean, std = self.stats_[(("bus", "ac_line", "bus"), "edge_attr")]
        data[("bus", "ac_line", "bus")].edge_attr = (
            data[("bus", "ac_line", "bus")].edge_attr - mean
        ) / std
        # transformer.edge_attr
        mean, std = self.stats_[(("bus", "transformer", "bus"), "edge_attr")]
        data[("bus", "transformer", "bus")].edge_attr = (
            data[("bus", "transformer", "bus")].edge_attr - mean
        ) / std

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