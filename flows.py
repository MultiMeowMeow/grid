import torch
from torch_scatter import scatter_sum


# ---------------------------------------------------------------------------
# Branch flow computations
# ---------------------------------------------------------------------------

def lflows(va, vm, edge_index, edge_attr):
    """
    Line flows in p.u.
    edge_attr = [angmin, angmax, b_fr, b_to, r, x, rate_a, rate_b, rate_c]
    Returns [E_line, 4] = [pf, qf, pt, qt]
    """
    V = vm * torch.exp(1j * va)
    fr, to = edge_index
    Vf, Vt = V[fr], V[to]

    b_fr, b_to, r, x = edge_attr[:, 2], edge_attr[:, 3], edge_attr[:, 4], edge_attr[:, 5]
    y = 1 / (r + 1j * x)

    Y_ff = y + 0.5j * b_fr
    Y_ft = -y
    Y_tf = -y
    Y_tt = y + 0.5j * b_to

    I_f = Y_ff * Vf + Y_ft * Vt
    I_t = Y_tf * Vf + Y_tt * Vt

    Sf = Vf * torch.conj(I_f)
    St = Vt * torch.conj(I_t)

    return torch.stack([Sf.real, Sf.imag, St.real, St.imag], dim=-1)


def tflows(va, vm, edge_index, edge_attr, shift_in_deg=False):
    """
    Transformer flows in p.u.
    edge_attr = [angmin, angmax, r, x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to]
    Returns [E_tr, 4] = [pf, qf, pt, qt]
    """
    V = vm * torch.exp(1j * va)
    fr, to = edge_index
    Vf, Vt = V[fr], V[to]

    r, x = edge_attr[:, 2], edge_attr[:, 3]
    tap, shift, b_fr, b_to = edge_attr[:, 7], edge_attr[:, 8], edge_attr[:, 9], edge_attr[:, 10]
    if shift_in_deg:
        shift = shift * (torch.pi / 180)

    y = 1 / (r + 1j * x)
    a = tap * torch.exp(1j * shift)

    Y_ff = (y + 0.5j * b_fr) / (tap * tap)
    Y_ft = -y / torch.conj(a)
    Y_tf = -y / a
    Y_tt = y + 0.5j * b_to

    I_f = Y_ff * Vf + Y_ft * Vt
    I_t = Y_tf * Vf + Y_tt * Vt

    Sf = Vf * torch.conj(I_f)
    St = Vt * torch.conj(I_t)

    return torch.stack([Sf.real, Sf.imag, St.real, St.imag], dim=-1)


# ---------------------------------------------------------------------------
# Small helpers: aggregate injections per bus
# ---------------------------------------------------------------------------

def bus_gen_injections(data, pg, qg, num_buses):
    """
    Aggregate generator injections to buses.
    pg, qg: [N_gen]
    Returns: Pg_bus, Qg_bus: [N_bus]
    """
    if ("generator", "generator_link", "bus") not in data.edge_types:
        return pg.new_zeros(num_buses), qg.new_zeros(num_buses)

    gen2bus = data[("generator", "generator_link", "bus")].edge_index[1]
    Pg_bus = scatter_sum(pg, gen2bus, dim=0, dim_size=num_buses)
    Qg_bus = scatter_sum(qg, gen2bus, dim=0, dim_size=num_buses)
    return Pg_bus, Qg_bus


def bus_load_demands(data, num_buses):
    """
    Aggregate load demands to buses.
    Returns: Pd_bus, Qd_bus: [N_bus]
    """
    if "load" not in data.node_types:
        return torch.zeros(num_buses, device=data["bus"].x.device), \
               torch.zeros(num_buses, device=data["bus"].x.device)

    load_x = data["load"].x  # [N_load, 2] = [pd, qd]
    if ("load", "load_link", "bus") not in data.edge_types:
        # if there are loads but no links, something is wrong
        raise RuntimeError("Missing (load, load_link, bus) edge type")
    load2bus = data[("load", "load_link", "bus")].edge_index[1]  # [N_load]

    Pd_bus = scatter_sum(load_x[:, 0], load2bus, dim=0, dim_size=num_buses)
    Qd_bus = scatter_sum(load_x[:, 1], load2bus, dim=0, dim_size=num_buses)
    return Pd_bus, Qd_bus


def bus_shunt_injections(data, vm, num_buses):
    """
    Compute shunt P/Q injections per bus:
        P_sh = g_sh * |V|^2
        Q_sh = -b_sh * |V|^2
    Returns: Psh_bus, Qsh_bus: [N_bus]
    """
    device = vm.device

    if "shunt" not in data.node_types:
        return torch.zeros(num_buses, device=device), torch.zeros(num_buses, device=device)

    if ("shunt", "shunt_link", "bus") not in data.edge_types:
        raise RuntimeError("Missing (shunt, shunt_link, bus) edge type")

    sh_x = data["shunt"].x  # [N_shunt, 2] = [bs, gs]
    bs, gs = sh_x[:, 0], sh_x[:, 1]

    sh2bus = data[("shunt", "shunt_link", "bus")].edge_index[1]  # [N_shunt]
    Bsh_bus = scatter_sum(bs, sh2bus, dim=0, dim_size=num_buses)
    Gsh_bus = scatter_sum(gs, sh2bus, dim=0, dim_size=num_buses)

    V2 = vm * vm  # [N_bus]

    Psh_bus = Gsh_bus * V2
    Qsh_bus = -Bsh_bus * V2
    return Psh_bus, Qsh_bus


def bus_branch_injections(num_buses, edge_index, flows):
    """
    Aggregate branch injections (lines or transformers) to buses.
    flows: [E, 4] = [pf, qf, pt, qt] in p.u.
    Returns: P_branch, Q_branch: [N_bus]
    """
    if flows is None or flows.numel() == 0:
        device = edge_index.device
        return torch.zeros(num_buses, device=device), torch.zeros(num_buses, device=device)

    fr, to = edge_index
    pf, qf, pt, qt = flows.unbind(-1)

    P_bus = scatter_sum(pf, fr, dim=0, dim_size=num_buses) + \
            scatter_sum(pt, to, dim=0, dim_size=num_buses)
    Q_bus = scatter_sum(qf, fr, dim=0, dim_size=num_buses) + \
            scatter_sum(qt, to, dim=0, dim_size=num_buses)
    return P_bus, Q_bus


# ---------------------------------------------------------------------------
# Equality constraints: nodal P/Q balance
# ---------------------------------------------------------------------------

def power_balance_residuals(data, va, vm, pg, qg, line_flows, xfmr_flows):
    """
    Build active/reactive power balance residuals at each bus:
        res_P = Pg - Pd - Psh - P_branch
        res_Q = Qg - Qd - Qsh - Q_branch
    Everything in p.u.
    """
    num_buses = va.shape[0]

    # Generator injections per bus
    Pg_bus, Qg_bus = bus_gen_injections(data, pg, qg, num_buses)

    # Load demands per bus
    Pd_bus, Qd_bus = bus_load_demands(data, num_buses)

    # Shunt injections per bus
    Psh_bus, Qsh_bus = bus_shunt_injections(data, vm, num_buses)

    # Branch injections from lines
    if ("bus", "ac_line", "bus") in data.edge_types and line_flows is not None:
        line_edge = data[("bus", "ac_line", "bus")].edge_index
        P_line, Q_line = bus_branch_injections(num_buses, line_edge, line_flows)
    else:
        device = va.device
        P_line = torch.zeros(num_buses, device=device)
        Q_line = torch.zeros(num_buses, device=device)

    # Branch injections from transformers
    if ("bus", "transformer", "bus") in data.edge_types and xfmr_flows is not None:
        xfmr_edge = data[("bus", "transformer", "bus")].edge_index
        P_xfmr, Q_xfmr = bus_branch_injections(num_buses, xfmr_edge, xfmr_flows)
    else:
        device = va.device
        P_xfmr = torch.zeros(num_buses, device=device)
        Q_xfmr = torch.zeros(num_buses, device=device)

    P_branch = P_line + P_xfmr
    Q_branch = Q_line + Q_xfmr

    res_P = Pg_bus - Pd_bus - Psh_bus - P_branch
    res_Q = Qg_bus - Qd_bus - Qsh_bus - Q_branch

    return res_P, res_Q


# ---------------------------------------------------------------------------
# Inequality constraints: thermal limits & angle limits
# ---------------------------------------------------------------------------

def thermal_violations(Smax, flows):
    """
    Compute thermal limit violations for a set of branches.
    edge_attr : [..., rate_a, ...] at rate_col_idx
    flows     : [E, 4] = [pf, qf, pt, qt] in p.u.
    Returns:
        v_from, v_to: [E] violations (>=0 means violation, 0 means satisfied)
    """
    if flows is None or flows.numel() == 0:
        return None, None

    pf, qf, pt, qt = flows.unbind(-1)

    Sf2_from = pf.pow(2) + qf.pow(2)
    Sf2_to   = pt.pow(2) + qt.pow(2)
    Smax2    = Smax.pow(2)

    g_from = Sf2_from - Smax2
    g_to   = Sf2_to   - Smax2

    v_from = torch.relu(g_from)
    v_to   = torch.relu(g_to)
    return v_from, v_to


def angle_violations(va, edge_index, angmin, angmax):
    """
    Compute angle difference violations:
        angmin <= va_f - va_t <= angmax
    Returns vector of violations >= 0 (0 if within bounds).
    """
    fr, to = edge_index
    ang = va[fr] - va[to]

    v_low  = torch.relu(angmin - ang)  # violation if ang < angmin
    v_high = torch.relu(ang - angmax)  # violation if ang > angmax
    return v_low + v_high


# ---------------------------------------------------------------------------
# Top-level constraint loss (CANOS + PINCO-style)
# ---------------------------------------------------------------------------

def constraint_losses(
    data,
    va, vm,          # [N_bus]
    pg, qg,          # [N_gen]
    w_eq=1.0,
    w_th=1.0,
    w_ang=0.0,
    shift_in_deg=False,
):
    """
    Compute a set of constraint losses:
        - power balance (equality)
        - branch thermal limits (inequality)
        - optional angle difference limits (inequality)
    All in p.u. space.

    Returns a dict with:
        {
            "eq":  loss_eq,
            "thermal": loss_th,
            "angle":   loss_ang,
            "total":   total_loss
        }
    """

    # 1) branch flows in p.u.
    line_flows = None
    xfmr_flows = None
    if ("bus", "ac_line", "bus") in data.edge_types:
        line_flows = lflows(va, vm, data[("bus", "ac_line", "bus")].edge_index, data[("bus", "ac_line", "bus")].edge_attr)
    if ("bus", "transformer", "bus") in data.edge_types:
        xfmr_flows = tflows(
            va, vm, data[("bus", "transformer", "bus")].edge_index, data[("bus", "transformer", "bus")].edge_attr, shift_in_deg=shift_in_deg)

    # 2) equality constraints: nodal P/Q balance
    res_P, res_Q = power_balance_residuals(data, va, vm, pg, qg, line_flows, xfmr_flows)
    loss_eq = (res_P.pow(2) + res_Q.pow(2)).mean()

    # 3) inequality: thermal limits (lines + transformers)
    thermal_viols = []
    if line_flows is not None:
        Smax = data[("bus", "ac_line", "bus")].edge_attr[:, 6]
        v_from_l, v_to_l = thermal_violations(Smax, line_flows)
        thermal_viols.extend([v_from_l, v_to_l])

    if xfmr_flows is not None:
        Smax = data[("bus", "transformer", "bus")].edge_attr[:, 4]
        v_from_t, v_to_t = thermal_violations(Smax, xfmr_flows)
        thermal_viols.extend([v_from_t, v_to_t])

    thermal_viols = [v for v in thermal_viols if v is not None]
    if thermal_viols:
        loss_th = torch.cat(thermal_viols).pow(2).mean()
    else:
        loss_th = torch.tensor(0.0, device=va.device)

    # 4) inequality: angle limits (optional)
    loss_ang = torch.tensor(0.0, device=va.device)
    angle_viols = []
    if w_ang != 0.0:
        # lines
        if ("bus", "ac_line", "bus") in data.edge_types:
            line_edge = data[("bus", "ac_line", "bus")].edge_index
            line_attr = data[("bus", "ac_line", "bus")].edge_attr
            angmin_l, angmax_l = line_attr[:, 0], line_attr[:, 1]
            v_ang_l = angle_violations(va, line_edge, angmin_l, angmax_l)
            angle_viols.append(v_ang_l)

        # transformers
        if ("bus", "transformer", "bus") in data.edge_types:
            xfmr_edge = data[("bus", "transformer", "bus")].edge_index
            xfmr_attr = data[("bus", "transformer", "bus")].edge_attr
            angmin_t, angmax_t = xfmr_attr[:, 0], xfmr_attr[:, 1]
            v_ang_t = angle_violations(va, xfmr_edge, angmin_t, angmax_t)
            angle_viols.append(v_ang_t)

        if angle_viols:
            loss_ang = torch.cat(angle_viols).pow(2).mean()
        else:
            loss_ang = torch.tensor(0.0, device=va.device)

    # 5) combine with weights
    total = w_eq * loss_eq + w_th * loss_th + w_ang * loss_ang

    return {
        "eq": loss_eq,
        "thermal": loss_th,
        "angle": loss_ang,
        "total": total,
    }
