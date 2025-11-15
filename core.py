# canos_hetero_processor_loopy.py
from typing import Optional, Literal, Tuple, Dict
import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.data import HeteroData
from utils import weight_init
from opf_normalization import OPFNormalizer

class MLP(nn.Module):
    """Linear -> LayerNorm -> ReLU -> Linear"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ProcessorLayer(nn.Module):

    PHYSICAL: Tuple[Tuple[str, str, str], ...] = (
        ('bus', 'ac_line',     'bus'),
        ('bus', 'transformer', 'bus'),
    )
    LINKS: Tuple[Tuple[str, str, str], ...] = (
        ('generator', 'generator_link', 'bus'),
        ('bus',       'generator_link', 'generator'),
        ('load',      'load_link',      'bus'),
        ('bus',       'load_link',      'load'),
        ('shunt',     'shunt_link',     'bus'),
        ('bus',       'shunt_link',     'shunt'),
    )
    NODE_TYPES: Tuple[str, ...] = ('bus', 'generator', 'load', 'shunt')

    def __init__(
        self,
        hidden_size: int,
        *,
        aggregation: Literal["sum", "mean"] = "sum",
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.H = hidden_size
        self.aggregation = aggregation
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        # Per-physical-relation edge MLPs (input=[v_src, v_dst, e])
        self.edge_mlps = nn.ModuleDict()
        for et in self.PHYSICAL:
            key = self._etype_key(et)
            self.edge_mlps[key] = MLP(3 * hidden_size, hidden_size)

        # Per-link-relation message MLPs (input=[v_src, v_dst])
        self.link_mlps = nn.ModuleDict()
        for et in self.LINKS:
            key = self._etype_key(et)
            self.link_mlps[key] = MLP(2 * hidden_size, hidden_size)

        # Per-node-type update MLPs (input=[v, agg_msg])
        self.node_mlps = nn.ModuleDict({
            ntype: MLP(2 * hidden_size, hidden_size)
            for ntype in self.NODE_TYPES
        })
        self.apply(weight_init)

    @staticmethod
    def _etype_key(etype: Tuple[str, str, str]) -> str:
        return "__".join(etype)  # e.g., "bus__ac_line__bus"

    def _agg(self, msg: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
        if self.aggregation == "mean":
            return scatter_mean(msg, index=index, dim=0, dim_size=dim_size)
        return scatter_sum(msg, index=index, dim=0, dim_size=dim_size)

    def forward(self, data: HeteroData) -> HeteroData:
        H = self.H
        # 1) Fetch node embeddings (with sanity checks)
        node_h: Dict[str, torch.Tensor] = {}
        for ntype in self.NODE_TYPES:
            h = data[ntype].h
            assert h.dim() == 2 and h.size(-1) == H, f"{ntype}.h must be [N,{H}]"
            node_h[ntype] = h

        # 2) Message buffers per node type
        msg_buf: Dict[str, torch.Tensor] = {nt: torch.zeros_like(node_h[nt]) for nt in self.NODE_TYPES}

        # 3) Physical relations: update edge_attr (residual) + send messages to dst
        for etype in self.PHYSICAL:
            key = self._etype_key(etype)
            src_t, _, dst_t = etype
            store = data[etype]
            edge_index = store.edge_index
            e = store.h

            src_idx, dst_idx = edge_index[0], edge_index[1]
            v_src = node_h[src_t][src_idx]
            v_dst = node_h[dst_t][dst_idx]

            inp = torch.cat([v_src, v_dst, e], dim=-1)
            delta_e = self.dropout(self.edge_mlps[key](inp))
            store.h = e + delta_e
            msg = self._agg(delta_e, dst_idx, dim_size=node_h[dst_t].size(0))
            msg_buf[dst_t] = msg_buf[dst_t] + msg

        # 4) Artificial links: message-only (no edge state)
        for etype in self.LINKS:
            key = self._etype_key(etype)
            src_t, _, dst_t = etype
            store = data[etype]
            edge_index = store.edge_index
            src_idx, dst_idx = edge_index[0], edge_index[1]

            v_src = node_h[src_t][src_idx]
            v_dst = node_h[dst_t][dst_idx]

            inp = torch.cat([v_src, v_dst], dim=-1)
            msg = self.dropout(self.link_mlps[key](inp))
            msg_buf[dst_t] = msg_buf[dst_t] + self._agg(msg, dst_idx, dim_size=node_h[dst_t].size(0))

        # 5) Node updates (residual per node type)
        for ntype in self.NODE_TYPES:
            v = node_h[ntype]
            vin = torch.cat([v, msg_buf[ntype]], dim=-1)
            dv = self.dropout(self.node_mlps[ntype](vin))
            data[ntype].h = v + dv

        return data



class OPFCore(nn.Module):

    NODE_FEATS = {
        "bus": 4,
        "generator": 11,
        "load": 2,
        "shunt": 2,
    }
    EDGE_FEATS = {
        ("bus", "ac_line", "bus"): 9,
        ("bus", "transformer", "bus"): 11,
    }
    BUS_TYPE = 4
    @staticmethod
    def _etype_key(et: Tuple[str, str, str]) -> str:
        return "__".join(et)

    @staticmethod
    def _strip_labels(proc: HeteroData) -> None:
        for ntype in proc.node_types:
            store = proc[ntype]
            if "y" in store:
                del store["y"]
        for etype in proc.edge_types:
            store = proc[etype]
            if "edge_label" in store:
                del store["edge_label"]

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        *,
        aggregation: Literal["sum", "mean"] = "sum",
    ):
        super().__init__()
        self.H = hidden_size
        self.norm = OPFNormalizer()

        # Encoders
        self.bus_type_emb = nn.Embedding(self.BUS_TYPE, hidden_size)
        self.enc_bus = MLP(3 + hidden_size, hidden_size)
        self.node_encoders = nn.ModuleDict({
            nt: MLP(self.NODE_FEATS[nt], hidden_size)
            for nt in self.NODE_FEATS if nt != "bus"
        })
        self.edge_encoders = nn.ModuleDict({
            self._etype_key(et): MLP(self.EDGE_FEATS[et], hidden_size)
            for et in self.EDGE_FEATS
        })
        self._edge_keys = {self._etype_key(et): et for et in self.EDGE_FEATS}

        # Processor stack
        self.layers = nn.ModuleList([
            ProcessorLayer(
                hidden_size=hidden_size, aggregation=aggregation
            )
            for _ in range(num_layers)
        ])

        self.dec_bus = MLP(hidden_size, 3)
        self.dec_gen = MLP(hidden_size, 2)

    def _encode(self, data: HeteroData, raw: HeteroData) -> HeteroData:
        data = self.norm.normalize(data) if self.norm is not None else data

        bx = data["bus"].x
        data["bus"].h = self.enc_bus(
            torch.cat([bx[:, (0, 2, 3)], self.bus_type_emb(bx[:, 1].long())], dim=-1)
        )

        for nt, enc in self.node_encoders.items():
            data[nt].h = enc(data[nt].x)

        for key, enc in self.edge_encoders.items():
            et = self._edge_keys[key]
            data[et].h = enc(data[et].edge_attr)
        return data

    def _decode(self, proc: HeteroData, raw: HeteroData) -> HeteroData:
        # Bus: va (angle, rad), vm (magnitude, p.u.)
        s, c, z_vm = torch.unbind(self.dec_bus(proc["bus"].h), dim=-1)
        norm = torch.sqrt(s * s + c * c).clamp_min(1e-8)
        va = torch.atan2(s / norm, c / norm)

        vmin, vmax = raw["bus"].x[:, 2:4]
        vm = vmin + torch.sigmoid(z_vm) * (vmax - vmin)
        bus_pred = torch.stack([va, vm], dim=-1)

        # Generator: pg, qg
        z_pg, z_qg = torch.unbind(self.dec_gen(proc["generator"].h), dim=-1)

        pmin, pmax = raw["generator"].x[:, 2:4]
        qmin, qmax = raw["generator"].x[:, 5:7]
        pg = pmin + torch.sigmoid(z_pg) * (pmax - pmin)
        qg = qmin + torch.sigmoid(z_qg) * (qmax - qmin)
        gen_pred = torch.stack([pg, qg], dim=-1)

        return bus_pred, gen_pred

    def forward(self, data: HeteroData) -> HeteroData:
        raw = data
        proc = data.clone()
        self._strip_labels(proc)
        proc = self._encode(proc, raw)

        for layer in self.layers:
            proc = layer(proc)

        bus_pred, gen_pred = self._decode(proc, raw)
        return bus_pred, gen_pred
