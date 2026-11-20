"""
engine.py - lumped, coupled town + data-centre water model.

No UI code in here: numpy and pandas only, so it can be reused from a notebook,
a script or another app.

Model in one paragraph
----------------------
A town is a single reservoir fed by a local stream, run on a daily time step.
Town demand is not constant (season, weekday, heat, growth).  A data centre
draws cooling water that depends on its cooling mode, cycles of concentration
and ambient temperature.  Water is allocated by a source selector
(reclaimed water -> groundwater in stress -> stream).  The reservoir level
feeds back into the data centre's cooling mode, and the town's delivered water
feeds back into the wastewater available for reuse.

    S(t+1) = S(t) + inflow - env. flow - town draw - data-centre draw - losses

Everything not stated by the user lives in ASSUMPTIONS / COSTS below and is
shown in the app so it can be challenged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Defaults (types matter: floats stay floats so the UI picks float widgets)
# --------------------------------------------------------------------------
DEFAULTS = {
    # Town
    "population": 100000,
    "lpcd": 135,              # litres per person per day (net use at the tap)
    "growth": 1.5,            # % population growth per year
    "leak_pct": 22,           # % of produced water lost in the network
    "storage_days": 150,      # reservoir size, days of town production
    "inflow_cover": 2.15,     # mean annual stream inflow / initial town net demand
    "env_pct": 15,            # % of inflow left in the stream
    "seasonal_amp": 15,       # % swing in demand between hot and cool season
    "hot_month": 5,
    "wet_month": 8,
    "mean_temp": 27.0,
    "temp_amp": 8.0,
    # Drought and operating rules
    "years": 10,
    "drought_start": 4,
    "drought_len": 2,
    "drought_cut": 30,        # % cut to stream inflow during drought
    "variability": 20,        # % year-to-year inflow variability (Monte Carlo)
    "restrict_trigger": 20,   # reservoir % at which town restrictions start
    "restrict_cut": 15,       # % demand cut during restriction
    "amber": 45,              # reservoir % for Amber
    "red": 30,                # reservoir % for Red
    # Data centre
    "it_mw": 100,
    "load_factor": 85,
    "base_cycles": 3.0,
    "hardness": 250,          # feed-water hardness, mg/L as CaCO3
    "dry_limit_temp": 33.0,   # above this, dry cooling cannot hold capacity
    "grid_l_kwh": 4.52,       # indirect water per kWh (LBNL US average; edit for your grid)
    "cap_pct": 5.0,           # policy cap: DC draw as % of drought-year supply
    # Source options
    "reuse_cap": 2.0,         # ML/day reclaimed water plant capacity
    "reuse_share": 70,        # max % of cooling makeup that can be reclaimed
    "gw_pump": 4.0,           # ML/day groundwater pumping capacity
    "gw_store": 1500,         # ML usable aquifer storage
    "gw_recharge": 900,       # ML/yr mean aquifer recharge
    "leak_cut": 40,           # % reduction in network leakage (option C)
    "pkg_cycles": 6.0,        # cycles of concentration in option C
    # Simulation
    "runs": 200,
    "seed": 7,
}

# Illustrative presets. Not calibrated to any real place.
PRESETS = {
    "Generic mid-size town": {},
    "Hot semi-arid city": {
        "population": 300000, "lpcd": 120, "leak_pct": 30, "storage_days": 110,
        "inflow_cover": 2.6, "mean_temp": 29.0, "temp_amp": 9.0, "drought_cut": 35,
        "it_mw": 150, "reuse_cap": 4.0, "gw_pump": 8.0, "gw_store": 2500, "gw_recharge": 1500,
        "hardness": 300,
    },
    "Small cool-climate town": {
        "population": 60000, "lpcd": 150, "leak_pct": 12, "storage_days": 200,
        "inflow_cover": 2.0, "mean_temp": 12.0, "temp_amp": 9.0, "hot_month": 7,
        "wet_month": 12, "it_mw": 60, "reuse_cap": 1.0, "gw_pump": 2.0,
        "gw_store": 800, "gw_recharge": 500, "dry_limit_temp": 30.0, "hardness": 180,
    },
}

ASSUMPTIONS = {
    "wue_ref": 0.83,                       # L/kWh IT at 3 cycles (about 2 ML/day per 100 MW)
    "mode_mult": [1.0, 0.45, 0.03],        # makeup water vs evaporative: evap, hybrid, dry
    "pue": [1.25, 1.32, 1.45],             # electricity overhead by mode
    "temp_coeff": 0.025,                   # evaporation change per degree C vs 26 C
    "hardness_limit": 1000.0,              # mg/L: max cooling-water hardness => max cycles
    "return_flow": 0.8,                    # share of town taps water returning as wastewater
    "res_loss": 0.0002,                    # reservoir evaporation, fraction of storage per day
    "reuse_tds": 700.0,                    # mg/L reclaimed water
    "reuse_tds_limit": 1000.0,             # mg/L accepted as cooling feed
    "feed_tds": 350.0,                     # mg/L fresh feed water
    "blowdown_tds_limit": 2000.0,          # mg/L discharge limit
    "shift_cut": 0.08,                     # workload shift when dry cooling is heat-limited
}

# Operating cost of each source, $/ML
COSTS = {"stream": 150.0, "reuse": 900.0, "gw": 350.0}
# Annualised programme costs, $ per year (scale with capacity where noted)
FIXED = {
    "reuse_per_mld": 350_000.0,   # per ML/day of reclaimed-water capacity
    "gw_per_mld": 112_500.0,      # per ML/day of pumping capacity
    "leak_programme": 350_000.0,
    "conditioning": 250_000.0,
}

# Technology levers for the recommender: $/kL of gross saving, kWh/kL energy
LEVERS = {
    "Higher cycles of concentration": {"unit_cost": 0.40, "kwh_kl": 0.0, "who": "Developer", "when": "Before commissioning"},
    "Treated wastewater reuse": {"unit_cost": 1.10, "kwh_kl": 1.2, "who": "Developer with utility", "when": "Before commissioning"},
    "Leak detection and smart metering": {"unit_cost": 0.35, "kwh_kl": 0.0, "who": "Utility, developer-funded", "when": "Within 12 months of approval"},
    "Hybrid cooling retrofit": {"unit_cost": 3.50, "kwh_kl": None, "who": "Developer", "when": "Design stage"},
    "Zero liquid discharge (ZLD)": {"unit_cost": 4.00, "kwh_kl": 15.0, "who": "Developer", "when": "Design stage"},
}

SCENARIOS = {
    "Baseline": dict(label="Baseline: local stream only", short="Baseline",
                     reuse=False, gw=False, leak=False, cyc=False, adaptive=False),
    "A": dict(label="A: Reclaimed water for cooling", short="A: Reclaimed",
              reuse=True, gw=False, leak=False, cyc=False, adaptive=False),
    "B": dict(label="B: Added groundwater source", short="B: Groundwater",
              reuse=False, gw=True, leak=False, cyc=False, adaptive=False),
    "C": dict(label="C: Combined package", short="C: Combined",
              reuse=True, gw=True, leak=True, cyc=True, adaptive=True),
}

TIER_NAMES = ["Green", "Amber", "Red"]
MODE_NAMES = ["Evaporative", "Hybrid", "Dry"]


# --------------------------------------------------------------------------
# Forcing: demand, temperature and inflow that are NOT constant
# --------------------------------------------------------------------------
def _ar1(rng, T, R, rho):
    z = rng.standard_normal((T, R))
    e = np.empty_like(z)
    e[0] = z[0]
    k = np.sqrt(1 - rho ** 2)
    for i in range(1, T):
        e[i] = rho * e[i - 1] + k * z[i]
    return e


def _forcing(P, R, seed):
    T = int(P["years"]) * 365
    Y = int(P["years"])
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    doy = t % 365
    yr = t // 365
    hot = (P["hot_month"] - 0.5) * 365 / 12
    wet = (P["wet_month"] - 0.5) * 365 / 12

    # Run 0 is the clean central case (no random noise) so the picture is readable.
    inflow_noise = np.exp(0.35 * _ar1(rng, T, R, 0.92) - 0.5 * 0.35 ** 2)
    inflow_noise[:, 0] = 1.0
    tnoise = 2.0 * _ar1(rng, T, R, 0.8)
    tnoise[:, 0] = 0.0
    dnoise = 0.03 * rng.standard_normal((T, R))
    dnoise[:, 0] = 0.0

    sig = P["variability"] / 100.0
    am = np.exp(rng.normal(-0.5 * sig ** 2, sig, size=(Y, R)))
    am[:, 0] = 1.0
    drought = np.ones(Y)
    s0 = int(P["drought_start"]) - 1
    drought[s0:s0 + int(P["drought_len"])] = 1 - P["drought_cut"] / 100.0
    am = am * drought[:, None]

    season_hot = np.cos(2 * np.pi * (doy - hot) / 365)
    season_wet = 1 + 0.8 * np.cos(2 * np.pi * (doy - wet) / 365)

    temp = P["mean_temp"] + P["temp_amp"] * season_hot[:, None] + tnoise
    weekly = 1 + 0.03 * np.cos(2 * np.pi * t / 7)
    dfac = ((1 + P["seasonal_amp"] / 100 * season_hot) * weekly)[:, None] * (1 + 0.012 * tnoise) * (1 + dnoise)
    pop = P["population"] * (1 + P["growth"] / 100) ** (t / 365)
    D = pop[:, None] * P["lpcd"] * dfac / 1e6                     # ML/day, net at the tap

    town0 = P["population"] * P["lpcd"] / 1e6                      # ML/day
    infl_annual = P["inflow_cover"] * town0 * 365                  # ML/yr
    inflow = infl_annual / 365 * season_wet[:, None] * am[yr] * inflow_noise
    recharge = P["gw_recharge"] / 365 * season_wet[:, None] * am[yr]
    return dict(T=T, Y=Y, yr=yr, temp=temp, D=D, inflow=inflow, recharge=recharge,
                infl_annual=infl_annual, town0=town0)


# --------------------------------------------------------------------------
# Core simulation (vectorised over Monte Carlo runs)
# --------------------------------------------------------------------------
def simulate(P, scen, runs=None, seed=None, dc_on=True):
    A = ASSUMPTIONS
    cfg = SCENARIOS[scen] if isinstance(scen, str) else scen
    R = int(runs if runs is not None else P["runs"])
    F = _forcing(P, R, P["seed"] if seed is None else seed)
    T, Y, yr = F["T"], F["Y"], F["yr"]

    leak0 = P["leak_pct"] / 100.0
    leak = leak0 * (1 - P["leak_cut"] / 100.0) if cfg["leak"] else leak0
    Smax = P["storage_days"] * F["town0"] / (1 - leak0)
    trig, amber, red = P["restrict_trigger"] / 100, P["amber"] / 100, P["red"] / 100
    env = P["env_pct"] / 100

    # Cycles of concentration are limited by feed-water hardness
    hard_eff = P["hardness"] * (0.5 if cfg["cyc"] else 1.0)
    cmax = A["hardness_limit"] / max(hard_eff, 1.0)
    c_target = P["pkg_cycles"] if cfg["cyc"] else P["base_cycles"]
    c_eff = float(max(1.5, min(c_target, cmax)))
    cyc_factor = (c_eff / (c_eff - 1)) / 1.5

    mode_mult = np.array(A["mode_mult"])
    pue = np.array(A["pue"])
    reuse_ok = cfg["reuse"] and A["reuse_tds"] <= A["reuse_tds_limit"]
    it_kw = P["it_mw"] * P["load_factor"] / 100 * 1000 if dc_on else 0.0

    S = np.full(R, 0.9 * Smax)
    Aq = np.full(R, float(P["gw_store"]) * 0.9)
    ww_prev = A["return_flow"] * F["D"][0] * 0.95

    def Z():
        return np.zeros((Y, R))
    agg = {k: Z() for k in ("short", "restr", "dc_fresh", "stream", "cost", "curtail", "indirect", "spill", "gw")}
    names = ("frac", "aq_frac", "deliv", "tier", "mode", "town_stream", "town_gw", "dc_stream",
             "dc_gw", "dc_reuse", "dc_need", "temp", "D", "restricted", "indirect", "unserved", "inflow")
    det = {k: np.zeros(T) for k in names}
    frac_all = np.zeros((T, R), dtype=np.float32)

    for i in range(T):
        y = yr[i]
        frac = S / Smax
        restricted = frac < trig
        Dn = F["D"][i]
        Dreq = Dn * np.where(restricted, 1 - P["restrict_cut"] / 100, 1.0)
        prod_need = Dreq / (1 - leak)
        temp = F["temp"][i]

        tier = (frac <= amber).astype(int) + (frac <= red).astype(int)
        mode = np.zeros(R, dtype=int)
        shift = np.zeros(R)
        if cfg["adaptive"] and dc_on:
            mode = np.where(tier >= 1, 1, 0)
            dry_ok = temp <= P["dry_limit_temp"]
            mode = np.where((tier == 2) & dry_ok, 2, mode)
            shift = np.where((tier == 2) & ~dry_ok, A["shift_cut"], 0.0)
        kwh_it = it_kw * (1 - shift) * 24
        tempfac = np.clip(1 + A["temp_coeff"] * (temp - 26), 0.7, 1.5)
        need = kwh_it * A["wue_ref"] * 1e-6 * cyc_factor * mode_mult[mode] * tempfac     # ML/day
        indirect = kwh_it * pue[mode] * P["grid_l_kwh"] * 1e-6                           # ML/day

        # Source selector, step 1: reclaimed water (depends on town wastewater)
        if reuse_ok:
            avail_re = np.minimum(P["reuse_cap"], ww_prev)
            ru = np.minimum(need * P["reuse_share"] / 100, avail_re)
        else:
            ru = np.zeros(R)
        dc_rem = need - ru

        # Step 2: groundwater tops up in stress (Amber or worse)
        tot = prod_need + dc_rem
        if cfg["gw"]:
            gwtot = np.where(frac < amber, np.minimum(np.minimum(P["gw_pump"], Aq), tot), 0.0)
        else:
            gwtot = np.zeros(R)
        sh = prod_need / np.maximum(tot, 1e-9)
        gw_t = gwtot * sh
        gw_d = gwtot - gw_t
        ts = prod_need - gw_t
        ds = dc_rem - gw_d

        # Step 3: stream reservoir. Option C protects households first.
        inflow_net = F["inflow"][i] * (1 - env)
        avail = S + inflow_net
        if cfg["adaptive"]:
            tg = np.minimum(ts, avail)
            dg = np.minimum(ds, avail - tg)
        else:
            req = ts + ds
            r = np.minimum(req, avail) / np.maximum(req, 1e-9)
            tg, dg = ts * r, ds * r
        sup = tg + dg
        loss = A["res_loss"] * S
        S_new = S + inflow_net - sup - loss
        spill = np.maximum(S_new - Smax, 0)
        S = np.clip(S_new, 0, Smax)
        Aq = np.clip(np.minimum(P["gw_store"], Aq + F["recharge"][i] - gwtot), 0, None)

        prod_deliv = gw_t + tg
        taps = prod_deliv * (1 - leak)
        deliv = np.minimum(taps / Dn, 1.0) * 100
        ww_prev = A["return_flow"] * taps
        dc_water = ru + gw_d + dg
        unserved = np.where(need > 1e-9, 1 - dc_water / np.maximum(need, 1e-9), 0.0)

        agg["short"][y] += prod_deliv < prod_need * 0.999
        agg["restr"][y] += restricted
        agg["dc_fresh"][y] += dg + gw_d
        agg["stream"][y] += sup
        agg["gw"][y] += gwtot
        agg["curtail"][y] += unserved > 0.05
        agg["indirect"][y] += indirect
        agg["spill"][y] += spill
        agg["cost"][y] += sup * COSTS["stream"] + ru * COSTS["reuse"] + gwtot * COSTS["gw"]
        frac_all[i] = S / Smax

        det["frac"][i] = S[0] / Smax
        det["aq_frac"][i] = Aq[0] / max(P["gw_store"], 1)
        det["deliv"][i] = deliv[0]
        det["tier"][i] = tier[0]
        det["mode"][i] = mode[0]
        det["town_stream"][i] = tg[0]
        det["town_gw"][i] = gw_t[0]
        det["dc_stream"][i] = dg[0]
        det["dc_gw"][i] = gw_d[0]
        det["dc_reuse"][i] = ru[0]
        det["dc_need"][i] = need[0]
        det["temp"][i] = temp[0]
        det["D"][i] = Dn[0]
        det["restricted"][i] = float(restricted[0])
        det["indirect"][i] = indirect[0]
        det["unserved"][i] = unserved[0]
        det["inflow"][i] = F["inflow"][i][0]

    fixed = 0.0
    if cfg["reuse"]:
        fixed += FIXED["reuse_per_mld"] * P["reuse_cap"]
    if cfg["gw"]:
        fixed += FIXED["gw_per_mld"] * P["gw_pump"]
    if cfg["leak"]:
        fixed += FIXED["leak_programme"]
    if cfg["cyc"]:
        fixed += FIXED["conditioning"]

    drought_supply = F["infl_annual"] * (1 - env) * (1 - P["drought_cut"] / 100)
    if cfg["gw"]:
        drought_supply += P["gw_recharge"] * (1 - P["drought_cut"] / 100)
    dy = [k for k in range(int(P["drought_start"]) - 1, int(P["drought_start"]) - 1 + int(P["drought_len"])) if k < Y]
    short_flag = agg["short"] > 0

    kpi = dict(
        shortfall_prob=float(short_flag.mean()),
        shortfall_prob_drought=float(short_flag[dy].mean()) if dy else float("nan"),
        restrict_days=float(agg["restr"].mean()),
        dc_fresh_ml_yr=float(agg["dc_fresh"].mean()),
        dc_draw_pct=float(agg["dc_fresh"].mean() / drought_supply * 100),
        stream_ml_yr=float(agg["stream"].mean()),
        curtail_days=float(agg["curtail"].mean()),
        op_cost_yr=float(agg["cost"].mean()),
        fixed_cost_yr=float(fixed),
        cost_yr=float(agg["cost"].mean() + fixed),
        indirect_ml_yr=float(agg["indirect"].mean()),
        drought_supply=float(drought_supply),
        min_frac_median=float(np.median(frac_all.min(axis=0))),
        c_eff=c_eff,
        cmax=float(cmax),
    )
    central = pd.DataFrame(det)
    central["year"] = np.arange(T) / 365
    band = pd.DataFrame({
        "p10": np.percentile(frac_all, 10, axis=1),
        "p50": np.percentile(frac_all, 50, axis=1),
        "p90": np.percentile(frac_all, 90, axis=1),
    })
    return dict(key=scen if isinstance(scen, str) else "custom", cfg=cfg, central=central,
                band=band, kpi=kpi, Smax=float(Smax), infl_annual=F["infl_annual"])


def verdict(kpi):
    p, d = kpi["shortfall_prob"], kpi["restrict_days"]
    if p > 0.25 or d > 45:
        return "Red"
    if p > 0.05 or d > 7:
        return "Amber"
    return "Green"


def run_all(P):
    """Run the reference case (no data centre) and the four source options."""
    res = {}
    for k in SCENARIOS:
        res[k] = simulate(P, k)
    nodc = simulate(P, "Baseline", runs=1, dc_on=False)
    res["NoDC"] = nodc
    base = res["Baseline"]["kpi"]
    for k, r in res.items():
        if k == "NoDC":
            continue
        kp = r["kpi"]
        kp["verdict"] = verdict(kp)
        avoided = (base["stream_ml_yr"] - kp["stream_ml_yr"]) * 1000.0     # kL per year
        extra = kp["cost_yr"] - base["cost_yr"]
        kp["avoided_kl_yr"] = avoided
        kp["extra_cost_yr"] = extra
        kp["cost_per_kl"] = extra / avoided if (k != "Baseline" and avoided > 1000) else float("nan")
    res["best"] = best_option(res)
    return res


def best_option(res):
    """Cheapest option that reaches Green; otherwise the one with the lowest shortfall probability."""
    cands = [k for k in ("A", "B", "C")]
    green = [k for k in cands if res[k]["kpi"]["verdict"] == "Green"]
    if green:
        return min(green, key=lambda k: res[k]["kpi"]["cost_yr"])
    return min(cands, key=lambda k: (res[k]["kpi"]["shortfall_prob"], res[k]["kpi"]["restrict_days"]))


# --------------------------------------------------------------------------
# Decision layer
# --------------------------------------------------------------------------
def recommend(P, res):
    """Rank technology levers by cost per NET litre (site saving minus power-linked water)."""
    A = ASSUMPTIONS
    base = res["Baseline"]
    c = base["central"]
    mk_yr = c["dc_need"].mean() * 365                       # ML/yr evaporative makeup
    leak0 = P["leak_pct"] / 100
    town_leak_yr = (c["D"].mean() / (1 - leak0) * leak0) * 365
    c0 = max(1.5, min(P["base_cycles"], A["hardness_limit"] / P["hardness"]))
    cn = max(1.5, min(P["pkg_cycles"], A["hardness_limit"] / (P["hardness"] * 0.5)))
    kwh_it_yr = P["it_mw"] * P["load_factor"] / 100 * 1000 * 24 * 365
    grid = P["grid_l_kwh"]

    rows = []

    def add(name, gross_ml, mwh, side):
        lv = LEVERS[name]
        gross_ml = max(gross_ml, 0.0)
        indirect = mwh * grid / 1000.0                      # ML/yr
        net = gross_ml - indirect
        cost = lv["unit_cost"] * gross_ml * 1000.0          # $/yr
        cpk = cost / (net * 1000.0) if net > 0.5 else float("inf")
        rows.append(dict(Lever=name, side=side, gross=gross_ml, indirect=indirect, net=net,
                         cost_yr=cost, cost_per_net_kl=cpk, who=lv["who"], when=lv["when"]))

    if cn > c0 + 0.01:
        g = mk_yr * (1 - (cn / (cn - 1)) / (c0 / (c0 - 1)))
        add("Higher cycles of concentration", g, 0.0, "Data centre")
    if A["reuse_tds"] <= A["reuse_tds_limit"]:
        g = min(P["reuse_cap"] * 365, mk_yr * P["reuse_share"] / 100)
        add("Treated wastewater reuse", g, g * LEVERS["Treated wastewater reuse"]["kwh_kl"], "Data centre")
    add("Leak detection and smart metering", town_leak_yr * 0.4, 0.0, "Town")
    extra_kwh = kwh_it_yr * (A["pue"][1] - A["pue"][0])
    add("Hybrid cooling retrofit", mk_yr * (1 - A["mode_mult"][1]), extra_kwh / 1000.0, "Data centre")
    add("Zero liquid discharge (ZLD)", mk_yr / c0 * 0.9, mk_yr / c0 * 0.9 * LEVERS["Zero liquid discharge (ZLD)"]["kwh_kl"], "Data centre")

    df = pd.DataFrame(rows).sort_values("cost_per_net_kl").reset_index(drop=True)
    draw = base["kpi"]["dc_fresh_ml_yr"]
    allowed = P["cap_pct"] / 100 * base["kpi"]["drought_supply"]
    gap = max(0.0, draw - allowed)
    # Town-side leak savings may cover at most half the gap: the data centre must
    # also cut its own draw, otherwise the conditions card would ask nothing of it.
    leak_cap = 0.5 * gap
    cum, sel, counted = 0.0, [], []
    for _, r in df.iterrows():
        if cum >= gap or not np.isfinite(r["cost_per_net_kl"]):
            sel.append(False)
            counted.append(0.0)
            continue
        credit = min(r["net"], leak_cap) if r["side"] == "Town" else r["net"]
        sel.append(True)
        counted.append(credit)
        cum += credit
    df["selected"] = sel
    df["counted"] = counted
    residual = draw - cum
    if residual <= allowed:
        supportable = float(P["it_mw"])
    else:
        supportable = float(P["it_mw"] * allowed / max(residual, 1e-9))
    reuse_row = df[(df.Lever == "Treated wastewater reuse") & df.selected]
    reuse_pct = float(P["reuse_share"]) if len(reuse_row) else 0.0
    cyc_row = df[(df.Lever == "Higher cycles of concentration") & df.selected]
    leak_row = df[(df.Lever == "Leak detection and smart metering") & df.selected]
    return dict(
        table=df, gap=gap, draw=draw, allowed=allowed, closed=cum >= gap, cum=cum,
        supportable_mw=supportable, reuse_pct=reuse_pct,
        cycles=float(cn) if len(cyc_row) else float(c0),
        leak_ml=float(leak_row["counted"].sum()) if len(leak_row) else 0.0,
        mk_yr=mk_yr,
    )


def quality_flags(P, scen_res):
    A = ASSUMPTIONS
    c_eff = scen_res["kpi"]["c_eff"]
    blow = A["feed_tds"] * c_eff
    rows = [
        ("Cooling blowdown salinity", f"{blow:,.0f} mg/L at {c_eff:.1f} cycles",
         f"limit {A['blowdown_tds_limit']:,.0f} mg/L", "Within limit" if blow <= A["blowdown_tds_limit"] else "Escalate to regulator: use ZLD or fewer cycles"),
        ("Reclaimed water quality", f"{A['reuse_tds']:,.0f} mg/L",
         f"limit {A['reuse_tds_limit']:,.0f} mg/L for cooling feed", "Within limit" if A["reuse_tds"] <= A["reuse_tds_limit"] else "Do not use"),
        ("Feed hardness and cycle ceiling", f"{P['hardness']} mg/L as CaCO3",
         f"max {scen_res['kpi']['cmax']:.1f} cycles", "Treatment needed above this" if scen_res["kpi"]["cmax"] < P["pkg_cycles"] else "No constraint"),
    ]
    return pd.DataFrame(rows, columns=["Check", "Value", "Limit", "Status"])


def monthly_view(central):
    """Monthly means of the central run (used for plots and the public report)."""
    m = (np.arange(len(central)) * 12 // 365).astype(int)
    g = central.groupby(m)
    out = g[["deliv", "frac", "town_stream", "town_gw", "dc_stream", "dc_gw", "dc_reuse", "dc_need", "indirect"]].mean()
    out["tier"] = g["tier"].max()
    out["mode"] = g["mode"].agg(lambda s: int(s.mode().iloc[0]))
    out["year"] = (out.index + 0.5) / 12
    return out


def sample_reports(central, seed=3):
    """Model-generated sample of resident reports, weekly, tied to delivered water."""
    rng = np.random.default_rng(seed)
    wk = (np.arange(len(central)) // 7).astype(int)
    d = central.groupby(wk)["deliv"].mean()
    types = ["Low pressure", "No water", "Discoloured water", "Dry well", "Night-time tanker traffic"]
    rows = []
    for w, v in d.items():
        lam = 0.4 + 0.55 * max(0.0, 100 - v)
        n = rng.poisson(lam)
        pw = np.array([0.45, 0.10 + (0.2 if v < 80 else 0), 0.2, 0.15, 0.1])
        pw = pw / pw.sum()
        for _ in range(n):
            rows.append(dict(Week=int(w), Zone=f"Zone {rng.integers(1, 5)}", Type=str(rng.choice(types, p=pw)), Source="Model sample"))
    return pd.DataFrame(rows, columns=["Week", "Zone", "Type", "Source"]), d


SLA_HOURS = {"No water": 24, "Low pressure": 72, "Discoloured water": 48, "Dry well": 72,
             "Night-time tanker traffic": 72}
