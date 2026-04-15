# pareto/dashboard.py
# Interactive Streamlit dashboard for exploring Pareto front results.
#
# Self-contained: no dependency on system/config.py.
#
# Usage:
#     cd pareto && streamlit run dashboard.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pareto_front import (
    OBJECTIVES,
    MAX_LOSS_DB_PER_CM,
    FOM_WEIGHTS,
    TARGETS,
    RESULTS_CSV,
    build_distributions,
    build_study,
)

st.set_page_config(page_title="Pareto Front – PIN PS Optimizer", layout="wide")


# ---------------------------------------------------------------------------
#  Data loading (cached — raw, unfiltered)
# ---------------------------------------------------------------------------
@st.cache_data
def load_raw():
    df = pd.read_csv(RESULTS_CSV)
    needed = list(OBJECTIVES) + ["v_pi_V"]
    df["reached_pi"] = df[needed].notna().all(axis=1)
    return df


def compute_pareto(valid_df):
    """Run Optuna study on valid rows and return set of Pareto row indices."""
    if valid_df.empty:
        return set()
    distributions = build_distributions()
    study = build_study(valid_df, distributions)
    return {t.number for t in study.best_trials}


def recalculate_cost(df, w_loss, w_vpil, t_loss, t_vpil):
    norm_loss = df[OBJECTIVES[1]] / t_loss
    norm_vpil = df[OBJECTIVES[0]] / t_vpil
    return w_loss * (norm_loss ** 2) + w_vpil * (norm_vpil ** 2)


def find_knee(pareto_df):
    if len(pareto_df) < 3:
        return None
    x = pareto_df[OBJECTIVES[0]].values
    y = pareto_df[OBJECTIVES[1]].values
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-30)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-30)
    p0 = np.array([x_norm[0], y_norm[0]])
    p1 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p1 - p0
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-30:
        return None
    line_unit = line_vec / line_len
    dists = np.abs(np.cross(line_unit, p0 - np.column_stack([x_norm, y_norm])))
    return int(np.argmax(dists))


def to_display_units(df: pd.DataFrame, cost_col: str = "cost") -> pd.DataFrame:
    d = pd.DataFrame()
    d["Sim ID"] = df["sim_id"].astype(int)
    d["w_r (nm)"] = (df["w_r"] * 1e9).round(1)
    d["h_si (nm)"] = (df["h_si"] * 1e9).round(1)
    d["Doping (cm⁻³)"] = df["doping"].map(lambda v: f"{v:.2e}")
    d["S (nm)"] = (df["S"] * 1e9).round(1)
    d["λ (nm)"] = (df["lambda"] * 1e9).round(1)
    d["Length (mm)"] = (df["length"] * 1e3).round(3)
    if "v_pi_V" in df.columns:
        d["V_π (V)"] = df["v_pi_V"].round(4)
    if OBJECTIVES[0] in df.columns:
        d["V_π·L (V·mm)"] = df[OBJECTIVES[0]].round(4)
    if OBJECTIVES[1] in df.columns:
        d["Loss (dB/cm)"] = df[OBJECTIVES[1]].round(2)
    if cost_col in df.columns:
        d["Cost"] = df[cost_col].round(4)
    if "reached_pi" in df.columns:
        d["Reached π"] = df["reached_pi"]
    return d


# ---------------------------------------------------------------------------
#  Load raw data
# ---------------------------------------------------------------------------
raw_df = load_raw()
n_total = len(raw_df)
n_reached_pi = int(raw_df["reached_pi"].sum())

# ---------------------------------------------------------------------------
#  Sidebar – Data Filters
# ---------------------------------------------------------------------------
st.sidebar.header("Data Filters")

only_reached_pi = st.sidebar.checkbox("Only sims that reached π", value=True)
max_loss = st.sidebar.slider(
    "Max Loss cutoff (dB/cm)", min_value=0.0,
    max_value=float(raw_df[OBJECTIVES[1]].max()) if raw_df[OBJECTIVES[1]].notna().any() else 200.0,
    value=float(MAX_LOSS_DB_PER_CM), step=5.0,
)

st.sidebar.caption(f"Total rows: {n_total} | Reached π: {n_reached_pi}")

valid = raw_df.copy()
if only_reached_pi:
    valid = valid[valid["reached_pi"]].copy()
valid = valid[valid[OBJECTIVES[1]].fillna(np.inf) <= max_loss].copy().reset_index(drop=True)

# Compute Pareto on the filtered valid set
pareto_indices = compute_pareto(valid)
valid["is_pareto"] = [i in pareto_indices for i in range(len(valid))]

# ---------------------------------------------------------------------------
#  Sidebar – Plot Filters
# ---------------------------------------------------------------------------
st.sidebar.header("Plot Filters")

loss_min, loss_max = float(valid[OBJECTIVES[1]].min()) if len(valid) else 0.0, float(valid[OBJECTIVES[1]].max()) if len(valid) else 1.0
vpil_min, vpil_max = float(valid[OBJECTIVES[0]].min()) if len(valid) else 0.0, float(valid[OBJECTIVES[0]].max()) if len(valid) else 1.0

loss_range = st.sidebar.slider(
    "Loss (dB/cm)", min_value=loss_min, max_value=loss_max,
    value=(loss_min, loss_max), step=0.5,
)
vpil_range = st.sidebar.slider(
    "V_π·L (V·mm)", min_value=vpil_min, max_value=vpil_max,
    value=(vpil_min, vpil_max), step=0.01,
)
show_all = st.sidebar.checkbox("Show non-Pareto points", value=True)
axes_from_zero = st.sidebar.checkbox("Axes start from 0", value=False)

# ---------------------------------------------------------------------------
#  Sidebar – Cost Function
# ---------------------------------------------------------------------------
st.sidebar.header("Cost Function")
st.sidebar.latex(r"C = w_\alpha \left(\frac{\alpha}{T_\alpha}\right)^2 + w_{V_\pi L}\left(\frac{V_\pi L}{T_{V_\pi L}}\right)^2")

w_loss = st.sidebar.slider("w_loss (loss weight)", 0.0, 1.0, float(FOM_WEIGHTS["loss"]), 0.05)
w_vpil = st.sidebar.slider("w_vpil (V_π·L weight)", 0.0, 1.0, float(FOM_WEIGHTS["vpil"]), 0.05)
t_loss = st.sidebar.slider("T_loss (dB/cm target)", 1.0, 50.0, float(TARGETS["loss"]), 1.0)
t_vpil = st.sidebar.slider("T_vpil (V·mm target)", 0.1, 5.0, float(TARGETS["vpil"]), 0.1)

cost_changed = (
    w_loss != FOM_WEIGHTS["loss"]
    or w_vpil != FOM_WEIGHTS["vpil"]
    or t_loss != TARGETS["loss"]
    or t_vpil != TARGETS["vpil"]
)

COST_COL = "custom_cost" if cost_changed else "cost"
if cost_changed:
    valid["custom_cost"] = recalculate_cost(valid, w_loss, w_vpil, t_loss, t_vpil)

# ---------------------------------------------------------------------------
#  Apply plot range filters
# ---------------------------------------------------------------------------
mask = (
    (valid[OBJECTIVES[1]] >= loss_range[0])
    & (valid[OBJECTIVES[1]] <= loss_range[1])
    & (valid[OBJECTIVES[0]] >= vpil_range[0])
    & (valid[OBJECTIVES[0]] <= vpil_range[1])
)
filtered = valid[mask]
filtered_pareto = filtered[filtered["is_pareto"]].sort_values(OBJECTIVES[0])
filtered_non_pareto = filtered[~filtered["is_pareto"]]

# ---------------------------------------------------------------------------
#  Title & metrics
# ---------------------------------------------------------------------------
st.title("PIN Phase Shifter – Pareto Front")
if cost_changed:
    st.caption("Cost recalculated with custom weights/targets")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Valid Sims", len(valid))
c2.metric("Pareto-Optimal", int(valid["is_pareto"].sum()))
c3.metric("Best Cost", f"{valid[COST_COL].min():.3f}" if len(valid) else "—")
c4.metric("Best V_π·L", f"{valid[OBJECTIVES[0]].min():.3f} V·mm" if len(valid) else "—")
c5.metric("Best Loss", f"{valid[OBJECTIVES[1]].min():.1f} dB/cm" if len(valid) else "—")


# ---------------------------------------------------------------------------
#  Hover helper
# ---------------------------------------------------------------------------
def _hover_text(row):
    return (
        f"<b>Sim {int(row['sim_id'])}</b><br>"
        f"V_π·L = {row[OBJECTIVES[0]]:.4f} V·mm<br>"
        f"Loss = {row[OBJECTIVES[1]]:.2f} dB/cm<br>"
        f"Cost = {row[COST_COL]:.4f}<br>"
        f"<br>"
        f"w_r = {row['w_r']*1e9:.1f} nm<br>"
        f"h_si = {row['h_si']*1e9:.1f} nm<br>"
        f"Doping = {row['doping']:.2e} cm⁻³<br>"
        f"S = {row['S']*1e9:.1f} nm<br>"
        f"λ = {row['lambda']*1e9:.1f} nm<br>"
        f"Length = {row['length']*1e3:.3f} mm<br>"
        f"V_π = {row['v_pi_V']:.4f} V"
    )


# ---------------------------------------------------------------------------
#  Pareto front scatter plot
# ---------------------------------------------------------------------------
fig_pareto = go.Figure()

if show_all and len(filtered_non_pareto) > 0:
    fig_pareto.add_trace(go.Scatter(
        x=filtered_non_pareto[OBJECTIVES[0]],
        y=filtered_non_pareto[OBJECTIVES[1]],
        mode="markers",
        marker=dict(size=7, color="rgba(180,180,180,0.5)",
                    line=dict(width=0.5, color="rgba(140,140,140,0.6)")),
        text=[_hover_text(r) for _, r in filtered_non_pareto.iterrows()],
        hoverinfo="text",
        name="Non-Pareto",
    ))

knee_row = None
if len(filtered_pareto) > 0:
    fig_pareto.add_trace(go.Scatter(
        x=filtered_pareto[OBJECTIVES[0]],
        y=filtered_pareto[OBJECTIVES[1]],
        mode="markers+lines",
        line=dict(width=1.5, color="rgba(99,110,250,0.3)", dash="dot"),
        marker=dict(
            size=10,
            color=filtered_pareto[COST_COL],
            colorscale="Viridis",
            colorbar=dict(title="Cost"),
            line=dict(width=1, color="white"),
        ),
        text=[_hover_text(r) for _, r in filtered_pareto.iterrows()],
        hoverinfo="text",
        name="Pareto Front",
    ))

    knee_idx = find_knee(filtered_pareto)
    if knee_idx is not None:
        knee_row = filtered_pareto.iloc[knee_idx]
        fig_pareto.add_trace(go.Scatter(
            x=[knee_row[OBJECTIVES[0]]],
            y=[knee_row[OBJECTIVES[1]]],
            mode="markers+text",
            marker=dict(size=18, color="red", symbol="star",
                        line=dict(width=1.5, color="white")),
            text=[f"Knee (Sim {int(knee_row['sim_id'])})"],
            textposition="top center",
            textfont=dict(size=13, color="red"),
            hoverinfo="text",
            hovertext=_hover_text(knee_row),
            name="Knee Point",
            showlegend=True,
        ))

fig_pareto.update_layout(
    xaxis_title="V_π·L  (V·mm)",
    yaxis_title="Optical Loss  (dB/cm)",
    template="plotly_white",
    height=520,
    margin=dict(l=60, r=30, t=40, b=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hoverlabel=dict(bgcolor="rgba(30,30,30,0.95)", font_size=13, font_color="white"),
    xaxis=dict(rangemode="tozero" if axes_from_zero else "normal"),
    yaxis=dict(rangemode="tozero" if axes_from_zero else "normal"),
)

st.plotly_chart(fig_pareto, width="stretch")

# ---------------------------------------------------------------------------
#  Knee point details
# ---------------------------------------------------------------------------
if knee_row is not None:
    st.subheader("Knee Point — Best Trade-off")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sim ID", int(knee_row["sim_id"]))
    k2.metric("V_π·L", f"{knee_row[OBJECTIVES[0]]:.4f} V·mm")
    k3.metric("Loss", f"{knee_row[OBJECTIVES[1]]:.2f} dB/cm")
    k4.metric("Cost", f"{knee_row[COST_COL]:.4f}")
    k5.metric("V_π", f"{knee_row['v_pi_V']:.4f} V")
    with st.expander("Full knee-point parameters"):
        kp = {
            "w_r": f"{knee_row['w_r']*1e9:.1f} nm",
            "h_si": f"{knee_row['h_si']*1e9:.1f} nm",
            "Doping": f"{knee_row['doping']:.2e} cm⁻³",
            "S": f"{knee_row['S']*1e9:.1f} nm",
            "λ": f"{knee_row['lambda']*1e9:.1f} nm",
            "Length": f"{knee_row['length']*1e3:.3f} mm",
        }
        st.json(kp)

# ---------------------------------------------------------------------------
#  Cost evolution chart
# ---------------------------------------------------------------------------
st.subheader("Cost vs. Simulation ID")

sorted_valid = valid.sort_values("sim_id")

sim_id_min, sim_id_max = int(sorted_valid["sim_id"].min()), int(sorted_valid["sim_id"].max())
cost_val_min, cost_val_max = float(sorted_valid[COST_COL].min()), float(sorted_valid[COST_COL].max())

cc1, cc2 = st.columns(2)
sim_id_range = cc1.slider(
    "Sim ID range", min_value=sim_id_min, max_value=sim_id_max,
    value=(sim_id_min, sim_id_max), step=1,
)
cost_range = cc2.slider(
    "Cost range", min_value=cost_val_min, max_value=cost_val_max,
    value=(cost_val_min, cost_val_max), step=0.01,
)

cost_mask = (
    (sorted_valid["sim_id"] >= sim_id_range[0])
    & (sorted_valid["sim_id"] <= sim_id_range[1])
    & (sorted_valid[COST_COL] >= cost_range[0])
    & (sorted_valid[COST_COL] <= cost_range[1])
)
cost_filtered = sorted_valid[cost_mask]
cost_running_best = cost_filtered[COST_COL].expanding().min()

fig_cost = go.Figure()
fig_cost.add_trace(go.Scatter(
    x=cost_filtered["sim_id"],
    y=cost_filtered[COST_COL],
    mode="markers",
    marker=dict(size=5, color="rgba(99,110,250,0.5)"),
    name="Cost per sim",
    hovertemplate="Sim %{x:.0f}<br>Cost = %{y:.4f}<extra></extra>",
))
fig_cost.add_trace(go.Scatter(
    x=cost_filtered["sim_id"],
    y=cost_running_best,
    mode="lines",
    line=dict(width=2.5, color="rgb(239,85,59)"),
    name="Running best",
    hovertemplate="Sim %{x:.0f}<br>Best so far = %{y:.4f}<extra></extra>",
))
fig_cost.update_layout(
    xaxis_title="Simulation ID",
    yaxis_title="Cost",
    template="plotly_white",
    height=360,
    margin=dict(l=60, r=30, t=20, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hoverlabel=dict(bgcolor="rgba(30,30,30,0.95)", font_size=13, font_color="white"),
)

st.plotly_chart(fig_cost, width="stretch")

# ---------------------------------------------------------------------------
#  Results table
# ---------------------------------------------------------------------------
table_view = st.radio(
    "Table view",
    ["Pareto-optimal only", "All valid (filtered)", "All results (raw)"],
    horizontal=True,
)

if table_view == "Pareto-optimal only":
    table_data = filtered_pareto
    st.subheader(f"Pareto-Optimal Designs ({len(table_data)})")
elif table_view == "All valid (filtered)":
    table_data = valid
    st.subheader(f"All Valid Sims ({len(table_data)})")
else:
    table_data = raw_df.sort_values("sim_id")
    st.subheader(f"All Results ({len(table_data)})")

display_df = to_display_units(table_data, cost_col=COST_COL)
st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
    height=min(40 * len(display_df) + 50, 600),
)

csv_bytes = display_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_bytes, "pareto_results.csv", "text/csv")
