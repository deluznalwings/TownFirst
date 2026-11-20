"""
app.py - Streamlit front end for the coupled town + data-centre water model.

Run:  streamlit run app.py
All calculations live in engine.py; this file only collects inputs and draws results.
"""
import datetime as dt
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import engine as E

st.set_page_config(page_title="Town and Data Centre Water Model", layout="wide")

# Streamlit renamed use_container_width -> width="stretch" in newer releases.
try:
    _VER = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _VER = (0, 0)
STRETCH = {"width": "stretch"} if _VER >= (1, 50) else {"use_container_width": True}

COL = {"Baseline": "#e06c75", "A": "#56b6c2", "B": "#e5c07b", "C": "#98c379", "NoDC": "#8b95a5"}
TIER_COL = {0: "#56d364", 1: "#e3b341", 2: "#f85149"}
VERDICT_COL = {"Green": "#56d364", "Amber": "#e3b341", "Red": "#f85149"}
SRC = [  # key in monthly/central frame, label, colour
    ("town_stream", "Town: stream", "#4aa8d8"),
    ("town_gw", "Town: groundwater", "#2d6a8f"),
    ("dc_stream", "Data centre: stream", "#e06c75"),
    ("dc_gw", "Data centre: groundwater", "#e5c07b"),
    ("dc_reuse", "Data centre: reclaimed", "#98c379"),
]
OPTION_KEYS = ["Baseline", "A", "B", "C"]

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
.kpi {border: 1px solid rgba(255,255,255,.10); border-radius: 10px; padding: 12px 14px;
      background: rgba(255,255,255,.03);}
.kpi .l {font-size: .78rem; opacity: .7;}
.kpi .v {font-size: 1.75rem; font-weight: 600; line-height: 1.25;}
.kpi .s {font-size: .75rem; opacity: .65;}
.verdict {padding: .55rem .9rem; border-radius: 8px; border-left: 6px solid;
          background: rgba(255,255,255,.04); margin-bottom: .7rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------------------------
for _k, _v in E.DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("preset", list(E.PRESETS)[0])
st.session_state.setdefault("sel", "Baseline")


def _apply_preset():
    vals = {**E.DEFAULTS, **E.PRESETS[st.session_state["preset"]]}
    for k, v in vals.items():
        st.session_state[k] = float(v) if isinstance(E.DEFAULTS[k], float) else int(v)
    st.session_state["_apply_now"] = True


def num(label, key, lo, hi, step, help=None, kind="slider"):
    """Widget whose numeric type follows DEFAULTS, so int/float never mix."""
    cast = float if isinstance(E.DEFAULTS[key], float) else int
    fn = st.slider if kind == "slider" else st.number_input
    return fn(label, cast(lo), cast(hi), step=cast(step), key=key, help=help)


st.sidebar.title("Inputs")
st.sidebar.selectbox("Case", list(E.PRESETS), key="preset", on_change=_apply_preset,
                     help="Illustrative starting points, not calibrated to any real place.")
run_clicked = st.sidebar.button("Run simulation", type="primary", **STRETCH)
note = st.sidebar.empty()

with st.sidebar.expander("Town water system", expanded=True):
    num("Population served", "population", 5000, 2000000, 5000, kind="input")
    num("Demand per person (L/day)", "lpcd", 60, 300, 5)
    num("Population growth (% per year)", "growth", 0.0, 4.0, 0.1)
    num("Network leakage (% of production)", "leak_pct", 5, 45, 1)
    num("Reservoir size (days of production)", "storage_days", 30, 400, 10)
    num("Stream yield cover", "inflow_cover", 1.2, 4.0, 0.05,
        help="Mean annual stream inflow divided by the town's net annual demand at the start.")
    num("Share of inflow left in the stream (%)", "env_pct", 0, 40, 5)

with st.sidebar.expander("Climate and demand pattern"):
    num("Seasonal demand swing (%)", "seasonal_amp", 0, 40, 1)
    num("Hottest month", "hot_month", 1, 12, 1)
    num("Wettest month", "wet_month", 1, 12, 1)
    num("Mean temperature (C)", "mean_temp", 5.0, 40.0, 0.5)
    num("Seasonal temperature swing (C)", "temp_amp", 0.0, 15.0, 0.5)

with st.sidebar.expander("Drought and operating rules"):
    num("Years simulated", "years", 5, 20, 1)
    num("Drought starts in year", "drought_start", 1, 15, 1)
    num("Drought length (years)", "drought_len", 1, 5, 1)
    num("Stream inflow cut in drought (%)", "drought_cut", 0, 60, 5)
    num("Year-to-year inflow variability (%)", "variability", 0, 40, 5)
    num("Restriction trigger (reservoir %)", "restrict_trigger", 5, 40, 1)
    num("Demand cut under restriction (%)", "restrict_cut", 5, 40, 1)
    num("Amber level (reservoir %)", "amber", 20, 80, 1)
    num("Red level (reservoir %)", "red", 10, 60, 1)

with st.sidebar.expander("Data centre"):
    num("IT load (MW)", "it_mw", 10, 500, 10)
    num("Average load factor (%)", "load_factor", 40, 100, 1)
    num("Cycles of concentration today", "base_cycles", 1.5, 8.0, 0.5)
    num("Feed-water hardness (mg/L as CaCO3)", "hardness", 50, 600, 10)
    num("Max air temperature for dry cooling (C)", "dry_limit_temp", 20.0, 45.0, 0.5)
    num("Grid water intensity (L/kWh)", "grid_l_kwh", 0.5, 10.0, 0.01,
        help="Water used by power generation per kWh. Default is the US average; use your grid's value.")
    num("Policy cap: share of drought-year supply (%)", "cap_pct", 1.0, 20.0, 0.5)

with st.sidebar.expander("Source options A, B, C"):
    num("Reclaimed water plant (ML/day)", "reuse_cap", 0.0, 15.0, 0.5)
    num("Max reclaimed share of cooling water (%)", "reuse_share", 10, 100, 5)
    num("Groundwater pumping (ML/day)", "gw_pump", 0.0, 30.0, 0.5)
    num("Usable aquifer storage (ML)", "gw_store", 100, 10000, 100)
    num("Aquifer recharge (ML/year)", "gw_recharge", 0, 6000, 100)
    num("Leak reduction in option C (%)", "leak_cut", 0, 80, 5)
    num("Cycles of concentration in option C", "pkg_cycles", 3.0, 10.0, 0.5)

with st.sidebar.expander("Simulation"):
    num("Monte Carlo runs", "runs", 50, 500, 50)
    num("Random seed", "seed", 0, 9999, 1, kind="input")

with st.sidebar.expander("Fixed assumptions (edit in engine.py)"):
    rows = [(k, str(v)) for k, v in E.ASSUMPTIONS.items()]
    rows += [(f"cost per ML: {k}", f"${v:,.0f}") for k, v in E.COSTS.items()]
    st.dataframe(pd.DataFrame(rows, columns=["Assumption", "Value"]), hide_index=True)

P = {k: st.session_state[k] for k in E.DEFAULTS}
if not (P["restrict_trigger"] < P["red"] < P["amber"]):
    st.sidebar.warning("Levels should satisfy restriction < red < amber.")

if "applied" not in st.session_state:
    st.session_state["applied"] = dict(P)
if run_clicked or st.session_state.pop("_apply_now", False):
    st.session_state["applied"] = dict(P)
AP = st.session_state["applied"]
if P != AP:
    note.caption("Inputs changed. Press Run simulation to update the results.")


@st.cache_data(show_spinner="Running simulations")
def run_all(params_json):
    return E.run_all(json.loads(params_json))


APJ = json.dumps(AP, sort_keys=True)
res = run_all(APJ)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def kpi_card(label, value, sub, color=None):
    style = f' style="color:{color}"' if color else ""
    return f'<div class="kpi"><div class="l">{label}</div><div class="v"{style}>{value}</div><div class="s">{sub}</div></div>'


def fmt_cpk(x):
    return "n/a" if x is None or not np.isfinite(x) else f"${x:,.2f}"


def headline(res, key):
    b, k = res["Baseline"]["kpi"], res[key]["kpi"]
    if key == "Baseline":
        s = (f"With only the local stream, households face restrictions on about {k['restrict_days']:.0f} days a year, "
             f"and {k['shortfall_prob']*100:.0f}% of simulated years contain at least one day when demand cannot be met. "
             f"The data centre takes {k['dc_draw_pct']:.1f}% of drought-year supply and is itself short of water on about "
             f"{k['curtail_days']:.0f} days a year.")
        best = res["best"]
        bk = res[best]["kpi"]
        if bk["verdict"] == "Green":
            s += f" Option {best} is the cheapest option that reaches Green."
        else:
            s += f" No option reaches Green; option {best} comes closest."
        return s
    s = (f"Against the baseline, {E.SCENARIOS[key]['short']} moves shortfall probability from {b['shortfall_prob']*100:.0f}% "
         f"to {k['shortfall_prob']*100:.0f}% and restriction days from {b['restrict_days']:.0f} to {k['restrict_days']:.0f} a year. "
         f"The data centre's share of drought-year supply goes from {b['dc_draw_pct']:.1f}% to {k['dc_draw_pct']:.1f}%.")
    if np.isfinite(k["cost_per_kl"]):
        s += (f" It costs about ${k['extra_cost_yr']/1e6:.1f}M more per year, which is "
              f"${k['cost_per_kl']:.2f} per kL of stream water avoided.")
    return s


def build_animation(res, key, P):
    cfg = E.SCENARIOS[key]
    c, b = res[key]["central"], res["Baseline"]["central"]
    T, Y = len(c), int(P["years"])
    M = Y * 12
    yrs = c["year"].values
    wk = np.arange(0, T, 7)
    ends = [min(T - 1, int(round((m + 1) * 365 / 12)) - 1) for m in range(M)]
    starts = [0] + [e + 1 for e in ends[:-1]]
    tot = c["town_stream"] + c["town_gw"] + c["dc_stream"] + c["dc_gw"] + c["dc_reuse"]
    ymax = max(1.0, float(tot.max()) * 1.2)
    cats = ["Reservoir", "Aquifer"] if cfg["gw"] else ["Reservoir"]

    def state(m):
        s, e = starts[m], ends[m]
        seg = c.iloc[s:e + 1]
        tier, mode = int(c["tier"].iloc[e]), int(c["mode"].iloc[e])
        lev = [c["frac"].iloc[e] * 100] + ([c["aq_frac"].iloc[e] * 100] if cfg["gw"] else [])
        cols = [TIER_COL[tier]] + (["#4aa8d8"] if cfg["gw"] else [])
        mix = [float(seg[k].mean()) for k, _, _ in SRC]
        keep = wk[wk <= e]
        txt = (f"Year {yrs[e]:.1f}  |  {E.TIER_NAMES[tier]}  |  cooling: {E.MODE_NAMES[mode].lower()}  |  "
               f"delivered: {seg['deliv'].mean():.0f}%")
        return dict(lev=lev, cols=cols, mix=mix, lx=yrs[keep], ly=c["frac"].values[keep] * 100,
                    mx=[yrs[e]], my=[c["frac"].iloc[e] * 100], txt=txt)

    s0 = state(0)
    fig = make_subplots(rows=1, cols=3, column_widths=[0.17, 0.13, 0.70], horizontal_spacing=0.07)
    fig.add_trace(go.Bar(x=cats, y=s0["lev"], marker_color=s0["cols"], showlegend=False,
                         text=[f"{v:.0f}%" for v in s0["lev"]], textposition="outside"), row=1, col=1)
    for (k, lab, colr), v in zip(SRC, s0["mix"]):
        fig.add_trace(go.Bar(x=["Supply"], y=[v], name=lab, marker_color=colr), row=1, col=2)
    fig.add_trace(go.Scatter(x=yrs[wk], y=b["frac"].values[wk] * 100, mode="lines", name="Baseline",
                             line=dict(color=COL["NoDC"], width=2, dash="dot")), row=1, col=3)
    fig.add_trace(go.Scatter(x=s0["lx"], y=s0["ly"], mode="lines", name=cfg["short"],
                             line=dict(color=COL[key], width=3)), row=1, col=3)
    fig.add_trace(go.Scatter(x=s0["mx"], y=s0["my"], mode="markers", showlegend=False,
                             marker=dict(color=COL[key], size=11, line=dict(color="white", width=1))), row=1, col=3)
    fig.add_trace(go.Scatter(x=[Y * 0.01], y=[104], mode="text", text=[s0["txt"]], textposition="middle right",
                             showlegend=False, textfont=dict(size=13)), row=1, col=3)

    d0 = int(P["drought_start"]) - 1
    fig.add_vrect(x0=d0, x1=min(Y, d0 + int(P["drought_len"])), row=1, col=3,
                  fillcolor="rgba(227,179,65,0.10)", line_width=0, annotation_text="drought",
                  annotation_position="bottom left")
    for r_, c_ in ((1, 1), (1, 3)):
        fig.add_hline(y=P["restrict_trigger"], line_dash="dash", line_color=TIER_COL[2], line_width=1, row=r_, col=c_)
        fig.add_hline(y=P["amber"], line_dash="dot", line_color=TIER_COL[1], line_width=1, row=r_, col=c_)
    fig.update_yaxes(range=[0, 118], title_text="Storage (%)", row=1, col=1)
    fig.update_yaxes(range=[0, ymax], title_text="ML per day", row=1, col=2)
    fig.update_yaxes(range=[0, 108], title_text="Reservoir level (%)", row=1, col=3)
    fig.update_xaxes(range=[0, Y], title_text="Year", row=1, col=3)

    frames, steps = [], []
    for m in range(M):
        s = state(m)
        data = [go.Bar(x=cats, y=s["lev"], marker_color=s["cols"], text=[f"{v:.0f}%" for v in s["lev"]],
                       textposition="outside")]
        data += [go.Bar(x=["Supply"], y=[v]) for v in s["mix"]]
        data += [go.Scatter(x=s["lx"], y=s["ly"]), go.Scatter(x=s["mx"], y=s["my"]), go.Scatter(x=[Y * 0.01], y=[104], text=[s["txt"]])]
        frames.append(go.Frame(name=str(m), data=data, traces=[0, 1, 2, 3, 4, 5, 7, 8, 9]))
        steps.append(dict(method="animate", label=f"{m // 12 + 1}" if m % 12 == 0 else "",
                          args=[[str(m)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))]))
    fig.frames = frames
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=470, barmode="stack", margin=dict(t=70, b=95, l=10, r=10),
        legend=dict(orientation="h", y=1.16, x=0),
        updatemenus=[dict(type="buttons", direction="right", x=0.0, y=-0.14, xanchor="left", yanchor="top",
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, dict(frame=dict(duration=90, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="Pause", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                                           transition=dict(duration=0))])])],
        sliders=[dict(active=0, steps=steps, x=0.17, len=0.83, y=-0.14, yanchor="top",
                      currentvalue=dict(prefix="Year ", visible=False), pad=dict(t=0, b=0))],
    )
    return fig


def delivered_chart(res):
    fig = go.Figure()
    for k in OPTION_KEYS + ["NoDC"]:
        mv = E.monthly_view(res[k]["central"])
        name = "Town only (no data centre)" if k == "NoDC" else E.SCENARIOS[k]["short"]
        fig.add_trace(go.Scatter(x=mv["year"], y=mv["deliv"], mode="lines+markers", name=name,
                                 line=dict(color=COL[k], width=1.6, dash="dot" if k == "NoDC" else "solid"),
                                 marker=dict(size=4)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=340, margin=dict(t=30, b=40, l=10, r=10), legend=dict(orientation="h", y=1.15, x=0),
                      yaxis=dict(title="Share of household demand delivered (%)", range=[max(0, min(
                          res[k]["central"]["deliv"].min() for k in OPTION_KEYS) - 5), 101]),
                      xaxis=dict(title="Year"))
    return fig


def level_chart(res, sel, P):
    fig = go.Figure()
    t = res["Baseline"]["central"]["year"]
    for k in OPTION_KEYS:
        fig.add_trace(go.Scatter(x=t, y=res[k]["central"]["frac"] * 100, mode="lines", name=E.SCENARIOS[k]["short"],
                                 line=dict(color=COL[k], width=2.5 if k == sel else 1.3), opacity=1 if k == sel else 0.6))
    fig.add_trace(go.Scatter(x=t, y=res["NoDC"]["central"]["frac"] * 100, mode="lines", name="Town only",
                             line=dict(color=COL["NoDC"], dash="dot", width=1.6)))
    fig.add_hline(y=P["restrict_trigger"], line_dash="dash", line_color=TIER_COL[2], line_width=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=330, margin=dict(t=30, b=40, l=10, r=10), legend=dict(orientation="h", y=1.15, x=0),
                      yaxis=dict(title="Reservoir level (%)", range=[0, 102]), xaxis=dict(title="Year"))
    return fig


def band_chart(res, sel):
    fig = go.Figure()
    t = res[sel]["central"]["year"]
    s, b = res[sel]["band"], res["Baseline"]["band"]
    fig.add_trace(go.Scatter(x=t, y=s["p90"] * 100, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=s["p10"] * 100, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(86,182,194,0.20)", name="10th to 90th percentile"))
    fig.add_trace(go.Scatter(x=t, y=s["p50"] * 100, mode="lines", line=dict(color=COL[sel], width=2.5), name="Median"))
    if sel != "Baseline":
        fig.add_trace(go.Scatter(x=t, y=b["p50"] * 100, mode="lines", line=dict(color=COL["Baseline"], dash="dot", width=1.5),
                                 name="Baseline median"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=300, margin=dict(t=30, b=40, l=10, r=10), legend=dict(orientation="h", y=1.15, x=0),
                      yaxis=dict(title="Reservoir level (%)", range=[0, 102]), xaxis=dict(title="Year"))
    return fig


def mix_chart(row):
    fig = go.Figure()
    for k, lab, colr in SRC:
        v = float(row[k])
        if v > 0.005:
            fig.add_trace(go.Bar(y=["Supply"], x=[v], name=lab, orientation="h", marker_color=colr,
                                 text=f"{v:.1f}", textposition="inside"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      barmode="stack", height=170, margin=dict(t=10, b=30, l=10, r=10),
                      legend=dict(orientation="h", y=-0.4), xaxis=dict(title="ML per day"), yaxis=dict(visible=False))
    return fig


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("Town and Data Centre Water Model")
st.caption("A coupled reservoir model: what a proposed data centre does to a town's water in a dry year, and what to do about it.")

sel = st.radio("Source option", OPTION_KEYS, format_func=lambda k: E.SCENARIOS[k]["label"], horizontal=True, key="sel")
R = res[sel]
kp, kb = R["kpi"], res["Baseline"]["kpi"]

tab_over, tab_cmp, tab_do, tab_com, tab_pub = st.tabs(
    ["Overview", "Compare options", "What must be done", "Community desk", "Public report"])

# ---- Overview -------------------------------------------------------------
with tab_over:
    vc = VERDICT_COL[kp["verdict"]]
    st.markdown(f'<div class="verdict" style="border-color:{vc}"><b style="color:{vc}">{kp["verdict"]}</b>'
                f' &nbsp; {E.SCENARIOS[sel]["label"]}</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Shortfall probability", f"{kp['shortfall_prob']*100:.0f}%",
                         f"years with an unmet day. Baseline {kb['shortfall_prob']*100:.0f}%"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Restriction days per year", f"{kp['restrict_days']:.0f}",
                         f"reservoir below {AP['restrict_trigger']}%. Baseline {kb['restrict_days']:.0f}"), unsafe_allow_html=True)
    over = kp["dc_draw_pct"] > AP["cap_pct"]
    c3.markdown(kpi_card("Data centre share of drought supply", f"{kp['dc_draw_pct']:.1f}%",
                         f"policy cap {AP['cap_pct']:.1f}%", TIER_COL[2] if over else TIER_COL[0]), unsafe_allow_html=True)
    c4.markdown(kpi_card("Cost per kL secured", fmt_cpk(kp["cost_per_kl"]),
                         "extra yearly cost per kL of stream water avoided" if sel != "Baseline" else "pick an option to compare"),
                unsafe_allow_html=True)
    st.write("")
    st.write(headline(res, sel))
    st.caption(f"Data centre short of water on about {kp['curtail_days']:.0f} days a year. Power-linked (indirect) water: "
               f"{kp['indirect_ml_yr']:,.0f} ML per year, reported separately and not part of the town's balance.")

    st.plotly_chart(build_animation(res, sel, AP), config={"displaylogo": False})
    st.caption("Central run, one frame per month. Bars show tanks and the month's supply mix; the line is the reservoir with "
               "the data centre; the dotted line is the baseline. Press Play.")
    st.markdown("**Share of household demand delivered, by month**")
    st.plotly_chart(delivered_chart(res), config={"displaylogo": False})

# ---- Compare --------------------------------------------------------------
with tab_cmp:
    rows = []
    for k in OPTION_KEYS:
        q = res[k]["kpi"]
        rows.append({"Option": E.SCENARIOS[k]["short"], "Verdict": q["verdict"],
                     "Shortfall probability": f"{q['shortfall_prob']*100:.0f}%",
                     "Restriction days/yr": f"{q['restrict_days']:.0f}",
                     "Data centre share of drought supply": f"{q['dc_draw_pct']:.1f}%",
                     "Data centre short of water (days/yr)": f"{q['curtail_days']:.0f}",
                     "Annual cost ($M)": f"{q['cost_yr']/1e6:.2f}",
                     "Cost per kL secured": fmt_cpk(q["cost_per_kl"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)
    best = res["best"]
    if res[best]["kpi"]["verdict"] == "Green":
        st.write(f"Best under current inputs: **{E.SCENARIOS[best]['label']}**, the cheapest option that reaches Green.")
    else:
        st.write(f"No option reaches Green under current inputs. Closest: **{E.SCENARIOS[best]['label']}**.")
    st.markdown("**Reservoir level, central run**")
    st.plotly_chart(level_chart(res, sel, AP), config={"displaylogo": False})
    st.markdown(f"**Uncertainty: {E.SCENARIOS[sel]['short']}, {AP['runs']} runs**")
    st.plotly_chart(band_chart(res, sel), config={"displaylogo": False})

# ---- What must be done ----------------------------------------------------
rec = E.recommend(AP, res)
with tab_do:
    st.markdown("**Approval conditions**")
    with st.container(border=True):
        if rec["gap"] <= 0:
            st.write(f"The data centre's draw ({kb['dc_draw_pct']:.1f}% of drought-year supply) is already within the "
                     f"{AP['cap_pct']:.1f}% cap. Supportable at {AP['it_mw']} MW with the operating rules below.")
        else:
            parts = []
            if rec["reuse_pct"] > 0:
                parts.append(f"reclaimed water supplies at least {rec['reuse_pct']:.0f}% of cooling makeup")
            if rec["cycles"] > AP["base_cycles"]:
                parts.append(f"cooling runs at {rec['cycles']:.0f} or more cycles of concentration")
            if rec["leak_ml"] > 0:
                parts.append(f"{rec['leak_ml']:,.0f} ML per year of network leak reduction is funded and verified")
            parts.append(f"cooling switches to hybrid below {AP['amber']}% reservoir level and to dry mode below {AP['red']}% "
                         "where air temperature allows")
            parts.append("the data centre is curtailed before households when supply is short")
            head = (f"Supportable at {rec['supportable_mw']:.0f} MW if " if rec["closed"] else
                    f"Supportable at no more than {rec['supportable_mw']:.0f} MW even if ")
            st.write(head + "; ".join(parts[:-1]) + f"; and {parts[-1]}.")
            st.caption(f"Baseline draw {rec['draw']:,.0f} ML/yr against an allowance of {rec['allowed']:,.0f} ML/yr "
                       f"({AP['cap_pct']:.1f}% of drought-year supply). Gap: {rec['gap']:,.0f} ML/yr.")

    st.markdown("**Technology ranking, by cost per net litre saved**")
    t = rec["table"]
    show = pd.DataFrame({
        "Rank": np.arange(1, len(t) + 1),
        "Lever": t["Lever"],
        "Applies to": t["side"],
        "Net saving (ML/yr)": t["net"].map(lambda v: f"{v:,.0f}"),
        "Power-linked water added (ML/yr)": t["indirect"].map(lambda v: f"{v:,.0f}"),
        "Cost per net kL": t["cost_per_net_kl"].map(lambda v: "n/a" if not np.isfinite(v) else f"${v:,.2f}"),
        "Annual cost ($M)": (t["cost_yr"] / 1e6).map(lambda v: f"{v:.2f}"),
        "In package": np.where(t["selected"], "Yes", "No"),
    })
    st.dataframe(show, hide_index=True, **STRETCH)
    st.caption("Net saving is site saving minus power-linked water. Savings are estimated one lever at a time and overlap; "
               "option C in the simulation tests the combined effect.")

    st.markdown("**Corrective action report**")
    if rec["gap"] <= 0 and kb["verdict"] == "Green":
        st.write("No corrective action required under current inputs.")
    else:
        prob = (f"Data centre draws {kb['dc_draw_pct']:.1f}% of drought-year supply (cap {AP['cap_pct']:.1f}%); "
                f"baseline shortfall probability {kb['shortfall_prob']*100:.0f}%")
        ca = []
        for _, r in t[t["selected"]].iterrows():
            ca.append({"What is wrong": prob, "Lever": r["Lever"],
                       "By how much": f"{r['counted']:,.0f} ML/yr net ({r['counted']/max(rec['draw'], 1)*100:.0f}% of draw)",
                       "By when": r["when"], "Responsible": r["who"]})
        ca.append({"What is wrong": "Reservoir enters Amber or Red", "Lever": "Cooling-mode switching",
                   "By how much": f"Site draw cut by {(1-E.ASSUMPTIONS['mode_mult'][1])*100:.0f}% (hybrid) or "
                                  f"{(1-E.ASSUMPTIONS['mode_mult'][2])*100:.0f}% (dry) while active",
                   "By when": "Automatic from first day of operation", "Responsible": "Data centre operator"})
        ca.append({"What is wrong": "Supply cannot meet demand", "Lever": "Curtail data centre before households",
                   "By how much": "Protects household supply first", "By when": "Written into approval",
                   "Responsible": "Regulator and utility"})
        st.dataframe(pd.DataFrame(ca), hide_index=True, **STRETCH)

    with st.expander("Escalation ladder"):
        lad = pd.DataFrame([
            ["Green", f"Reservoir above {AP['amber']}%", "Evaporative cooling. Report daily draw.",
             "Publish the public dashboard weekly.", "Routine audit of reported draw."],
            ["Amber", f"Reservoir {AP['red']}% to {AP['amber']}%", "Switch to hybrid. Use reclaimed water first.",
             "Bring groundwater top-up online. Notify residents.", "Request draw data. Review complaints."],
            ["Red", f"Reservoir below {AP['red']}%", "Dry mode where air temperature allows, otherwise hybrid plus workload shifting.",
             f"Prepare restrictions starting at {AP['restrict_trigger']}%. Protect households first.",
             "Order curtailment of non-essential draw."],
            ["Restriction", f"Reservoir below {AP['restrict_trigger']}%", "Curtail to minimum cooling water.",
             f"Impose a {AP['restrict_cut']}% demand cut.", "Daily public reporting."],
        ], columns=["Level", "Trigger", "Data centre", "Utility", "Regulator"])
        st.dataframe(lad, hide_index=True, **STRETCH)
        st.caption("Also escalate to the regulator when a complaint cluster appears in the Community desk, "
                   "or when reported and modelled draw differ by more than 15%.")
    with st.expander("Quality flags"):
        st.dataframe(E.quality_flags(AP, R), hide_index=True, **STRETCH)

# ---- Community desk -------------------------------------------------------
with tab_com:
    st.session_state.setdefault("reports", [])
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown("**Report a water problem**")
        with st.form("report_form", clear_on_submit=True):
            zone = st.selectbox("Zone", [f"Zone {i}" for i in range(1, 5)])
            typ = st.selectbox("What are you seeing?", list(E.SLA_HOURS))
            details = st.text_area("Details (optional)", max_chars=300)
            sent = st.form_submit_button("Submit report")
        if sent:
            now = dt.datetime.now()
            st.session_state["reports"].append({
                "time": now, "Zone": zone, "Type": typ, "Details": details,
                "due": now + dt.timedelta(hours=E.SLA_HOURS[typ])})
            st.success(f"Report logged. Response due within {E.SLA_HOURS[typ]} hours.")
    with right:
        st.markdown("**Public log**")
        reps = st.session_state["reports"]
        if reps:
            log = pd.DataFrame([{"Received": r["time"].strftime("%d %b %H:%M"), "Zone": r["Zone"], "Type": r["Type"],
                                 "Respond by": r["due"].strftime("%d %b %H:%M"), "Status": "Open"} for r in reps[::-1]])
            st.dataframe(log, hide_index=True, **STRETCH)
        else:
            st.caption("No reports yet. Reports are kept for this session only.")

    thr = st.number_input("Cluster threshold (reports per zone in 7 days)", 2, 50, 5)
    week_ago = dt.datetime.now() - dt.timedelta(days=7)
    recent = [r["Zone"] for r in st.session_state["reports"] if r["time"] >= week_ago]
    hot = [z for z in sorted(set(recent)) if recent.count(z) >= thr]
    if hot:
        msg = f"Complaint cluster in {', '.join(hot)}."
        if kp["verdict"] == "Green":
            st.warning(msg + " The model says Green, so flag the model for review.")
        else:
            st.warning(msg + f" Consistent with the model verdict of {kp['verdict']}. Route to the utility.")

    st.markdown("**Early warning from model-generated sample reports**")
    if st.toggle("Show sample reports for the selected option"):
        rep, wkd = E.sample_reports(R["central"])
        counts = rep.groupby("Week").size().reindex(wkd.index, fill_value=0)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=wkd.index / 52, y=counts.values, name="Reports per week", marker_color="#4aa8d8"))
        fig.add_trace(go.Scatter(x=wkd.index / 52, y=wkd.values, name="Delivered (%)", mode="lines",
                                 line=dict(color=COL[sel], width=2)), secondary_y=True)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=300, margin=dict(t=30, b=40, l=10, r=10), legend=dict(orientation="h", y=1.15, x=0),
                          xaxis=dict(title="Year"))
        fig.update_yaxes(title_text="Reports", secondary_y=False)
        fig.update_yaxes(title_text="Delivered (%)", range=[0, 102], secondary_y=True)
        st.plotly_chart(fig, config={"displaylogo": False})
        st.caption("Sample reports are generated from the model for demonstration. In use, the same chart runs on the real log, "
                   "so a rise in reports while the model shows Green points to something the model is missing.")

    st.markdown("**Audit check: reported against modelled draw**")
    modelled = kp["dc_fresh_ml_yr"] / 365
    a1, a2 = st.columns(2)
    reported = a1.number_input("Draw reported by the operator (ML/day, fresh water)", 0.0, 100.0,
                               round(float(modelled), 2), 0.05,
                               key=f"rep_{sel}_{abs(hash(APJ)) % 10**6}")
    a2.metric("Modelled average fresh draw (ML/day)", f"{modelled:.2f}")
    mis = (reported - modelled) / max(modelled, 1e-9) * 100
    if abs(mis) > 15:
        st.error(f"Mismatch of {mis:+.0f}%. Flag to the regulator.")
    else:
        st.write(f"Difference of {mis:+.0f}%, within the 15% tolerance.")

# ---- Public report --------------------------------------------------------
with tab_pub:
    mv = E.monthly_view(R["central"])
    view = st.radio("Show", ["Typical month", "Worst month"], horizontal=True)
    if view == "Worst month":
        row = mv.loc[mv["deliv"].idxmin()]
    else:
        row = mv.iloc[(mv["deliv"] - mv["deliv"].median()).abs().argmin()]
    tier = int(row["tier"])
    p1, p2, p3 = st.columns(3)
    p1.markdown(kpi_card("Stress level", E.TIER_NAMES[tier], f"reservoir {row['frac']*100:.0f}% full", TIER_COL[tier]),
                unsafe_allow_html=True)
    p2.markdown(kpi_card("Cooling mode", E.MODE_NAMES[int(row["mode"])], "data centre"), unsafe_allow_html=True)
    p3.markdown(kpi_card("Households served", f"{row['deliv']:.0f}%", "of normal demand delivered"), unsafe_allow_html=True)
    st.plotly_chart(mix_chart(row), config={"displaylogo": False})

    story = (
        f"This simulation covers a town of {AP['population']:,} people and a {AP['it_mw']} MW data centre over {AP['years']} years, "
        f"including a {AP['drought_len']}-year drought from year {AP['drought_start']} that cuts stream inflow by {AP['drought_cut']}%.\n\n"
        f"Under {E.SCENARIOS[sel]['label'].lower()}, households face restrictions on about {kp['restrict_days']:.0f} days a year, and "
        f"in {kp['shortfall_prob']*100:.0f}% of simulated years there is at least one day when supply cannot meet demand. "
        f"The data centre takes {kp['dc_draw_pct']:.1f}% of the water available in a drought year, against a policy cap of {AP['cap_pct']:.1f}%.\n\n"
        f"Overall verdict: {kp['verdict']}. This is a screening model with simplified assumptions, not a forecast. "
        "Every input is listed in the sidebar and can be changed and re-run."
    )
    st.write(story)
    d1, d2 = st.columns(2)
    d1.download_button("Download report (text)", story, file_name="water_report.txt", **STRETCH)
    d2.download_button("Download monthly results (CSV)", mv.to_csv(), file_name="monthly_results.csv", **STRETCH)
