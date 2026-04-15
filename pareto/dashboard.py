# pareto/dashboard.py
# Interactive Streamlit dashboard for exploring Pareto front results.
#
# Self-contained: no dependency on system/config.py.
#
# Usage:
#     cd pareto && streamlit run dashboard.py

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pareto_front import (
    OBJECTIVES,
    MAX_LOSS_DB_PER_CM,
    FOM_WEIGHTS,
    TARGETS,
    RUN_FILES,
    build_distributions,
    build_study,
)

RUN_COLORS = {"Run 1": "rgb(99,110,250)", "Run 2": "rgb(239,85,59)"}
RUN_SYMBOLS = {"Run 1": "circle", "Run 2": "diamond"}

PARAM_COLS = ["w_r", "h_si", "doping", "S", "lambda", "length"]

HELP_PARETO = """
כל נקודה היא סימולציה אחת. ציר X הוא **V_π·L** (נמוך = טוב), ציר Y הוא **הפסד אופטי** (נמוך = טוב).

הנקודות הצבעוניות על קו מקווקו הן **חזית פארטו** — עיצובים שאי אפשר לשפר ביעד אחד בלי לפגוע בשני.

הנקודות האפורות הן עיצובים נשלטים (לא על החזית).

ה**כוכב האדום** הוא **נקודת הברך (Knee)** — פשרה גיאומטרית בין שני היעדים (מרחק מקסימלי מקו שמחבר את קצוות החזית במרחב מנורמל).
"""

HELP_COST = """
**עלות** כפונקציה של **מספר הסימולציה** (סדר הריצה).

הנקודות הן עלות לכל סימולציה; **הקו** הוא **running best** — העלות הנמוכה ביותר שהושגה עד אותה נקודה בזמן. ירידה בקו משמעותה שיפור מצטבר.

במצב **Both**, לכל ריצה יש צבע וסימבול נפרדים, וקו running-best לכל ריצה.
"""

HELP_LHS_BO = """
השוואה בין שלב **LHS** (Latin Hypercube Sampling — דגימה ראשונית) לשלב **BO** (Bayesian Optimization).

**Box plot**: קו באמצע = חציון, הקופסה = רבעון 25%–75%, שפם = טווח טיפוסי, נקודות = חריגים.

אם **BO** משפר את **LHS**, תצפה לקופסאות נמוכות יותר (לעלות ול־V_π·L) או לפחות הפסד — בהתאם ליעד.

החלוקה ל־LHS/BO מבוססת על **מספר הסימולציה** ועל סף שנקבע בסרגל הצד (ברירת מחדל: 60).
"""

HELP_PARALLEL = """
**קואורדינטות מקבילות**: כל **קו** הוא סימולציה אחת שעוברת דרך כל הצירים (פרמטרים ויעדים).

ה**צבע** נקבע לפי הבחירה (עלות / V_π·L / הפסד) ומסייע לזהות אילו שילובי פרמטרים מתאימים לערכים טובים.

**Doping** מוצג כ־log₁₀(ריכוז) כדי שהטווח הרחב יהיה קריא.

ניתן לגרור טווחים על כל ציר בגרף (Plotly) כדי לסנן קווים.

סמן **רק פארטו** כדי להתמקד בעיצובים לא נשלטים.
"""

HELP_HEATMAP = """
**מפת חום של קורלציה** בין **פרמטרי הקלט** לבין **היעדים** (V_π·L, הפסד, עלות).

**כחול** ≈ קורלציה חיובית (כשהפרמטר עולה, היעד נוטה לעלות). **אדום** ≈ קורלציה שלילית.

ערכים קרובים ל־**0** אומרים שאין קשר ליניארי חזק (ייתכנו קשרים לא ליניאריים).

המספרים על התאים הם מקדמי **פירסון** על סט הנקודות המסונן (valid).
"""

HELP_DIST = """
**היסטוגרמות** משוות את **התפלגות כל פרמטר** בין עיצובים **פארטו** (כחול) לבין **שאר העיצובים** (אפור).

אם ההתפלגות הכחולה **מרוכזת** באזור מסוים, זה רמז שאזור זה במרחב הפרמטרים מועדף לעיצובים על חזית פארטו.

ל־**doping** ציר X לוגריתמי (בסיס 10) לקריאות.
"""

HELP_KNEE = """
**נקודת הברך** — פשרה בין שני היעדים: נקודה על חזית פארטו שבה **שיפור** באחד מהיעדים דורש **תשלום גבוה** בשני.

החישוב: נרמול שני הצירים ל־[0,1], קו בין שני הקצוות של חזית הפארטו, ובחירת הנקודה עם **מרחק אנכי מקסימלי** מהקו.
"""

HELP_TABLE = """
**טבלת תוצאות** עם יחידות קריאות.

- **Pareto-optimal only** — רק עיצובים לא נשלטים (לפי הפילטרים והחישוב הנוכחי).
- **All valid** — כל הסימולציות שעוברות את מסנן ההפסד ותנאי ה־π.
- **All results (raw)** — כל השורות מהקובץ, כולל כשלונות.

ניתן למיין לפי עמודה ולהוריד **CSV**.
"""

st.set_page_config(page_title="Pareto Front – PIN PS Optimizer", layout="wide")


def section_header(title: str, help_text: str) -> None:
    """Subheader with a ? popover for explanation."""
    col_title, col_help = st.columns([20, 1])
    with col_title:
        st.subheader(title)
    with col_help:
        with st.popover("?"):
            st.markdown(help_text)


def param_axis_label(name: str) -> str:
    return {
        "w_r": "w_r (nm)",
        "h_si": "h_si (nm)",
        "doping": "Doping (cm⁻³)",
        "S": "S (nm)",
        "lambda": "λ (nm)",
        "length": "Length (mm)",
    }.get(name, name)


def target_axis_label(col: str) -> str:
    if col == OBJECTIVES[0]:
        return "V_π·L (V·mm)"
    if col == OBJECTIVES[1]:
        return "Loss (dB/cm)"
    if col in ("cost", "custom_cost"):
        return "Cost"
    return col


def build_phase_boxplot(df: pd.DataFrame, col: str, y_title: str, lhs_count: int) -> go.Figure:
    fig = go.Figure()
    for phase, color in [("LHS", "rgb(99,110,250)"), ("BO", "rgb(0,186,56)")]:
        subset = df[df["sim_id"] <= lhs_count] if phase == "LHS" else df[df["sim_id"] > lhs_count]
        if subset.empty:
            continue
        fig.add_trace(
            go.Box(
                y=subset[col],
                name=phase,
                marker_color=color,
                boxmean=True,
            )
        )
    fig.update_layout(
        yaxis_title=y_title,
        template="plotly_white",
        height=280,
        margin=dict(l=50, r=20, t=30, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
#  Data loading (cached — raw, unfiltered)
# ---------------------------------------------------------------------------
@st.cache_data
def load_raw(run_key: str):
    if run_key == "Both":
        parts = []
        for name, path in RUN_FILES.items():
            d = pd.read_csv(path)
            d["run"] = name
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(RUN_FILES[run_key])
        df["run"] = run_key
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
    df = df.reset_index(drop=True)
    d = pd.DataFrame()
    if "run" in df.columns:
        d["Run"] = df["run"]
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
#  Sidebar – Run selection
# ---------------------------------------------------------------------------
st.sidebar.header("Run Selection")
run_choice = st.sidebar.radio(
    "Show results from",
    list(RUN_FILES.keys()) + ["Both"],
    index=0,
)

# ---------------------------------------------------------------------------
#  Load raw data
# ---------------------------------------------------------------------------
raw_df = load_raw(run_choice)
n_total = len(raw_df)
n_reached_pi = int(raw_df["reached_pi"].sum())

# ---------------------------------------------------------------------------
#  Sidebar – Data Filters
# ---------------------------------------------------------------------------
st.sidebar.header("Data Filters")

only_reached_pi = st.sidebar.checkbox("Only sims that reached π", value=True)
max_loss = st.sidebar.number_input(
    "Max Loss cutoff (dB/cm)", min_value=0.0,
    value=float(MAX_LOSS_DB_PER_CM), step=5.0, format="%.1f",
)

run_breakdown = ""
if run_choice == "Both":
    counts = raw_df["run"].value_counts().to_dict()
    run_breakdown = "  ".join(f"{k}: {counts.get(k, 0)}" for k in RUN_FILES)
    st.sidebar.caption(f"Total rows: {n_total} | Reached π: {n_reached_pi}  ({run_breakdown})")
else:
    st.sidebar.caption(f"{run_choice}: {n_total} rows | Reached π: {n_reached_pi}")

all_with_objectives = raw_df.copy()
if only_reached_pi:
    all_with_objectives = all_with_objectives[all_with_objectives["reached_pi"]].copy()
else:
    needed = list(OBJECTIVES) + ["v_pi_V"]
    all_with_objectives = all_with_objectives[all_with_objectives[needed].notna().all(axis=1)].copy()
valid = all_with_objectives[all_with_objectives[OBJECTIVES[1]] <= max_loss].copy().reset_index(drop=True)

# Compute Pareto on the filtered valid set
pareto_indices = compute_pareto(valid)
valid["is_pareto"] = [i in pareto_indices for i in range(len(valid))]

# ---------------------------------------------------------------------------
#  Sidebar – LHS / BO boundary
# ---------------------------------------------------------------------------
st.sidebar.header("LHS / BO Boundary")
lhs_count = st.sidebar.number_input(
    "LHS sample count (per run)",
    min_value=1,
    value=60,
    step=1,
    help="Sims with sim_id ≤ this value are tagged LHS; above = BO.",
)

# ---------------------------------------------------------------------------
#  Sidebar – Plot Filters
# ---------------------------------------------------------------------------
st.sidebar.header("Plot Filters")

loss_min = float(valid[OBJECTIVES[1]].min()) if len(valid) else 0.0
loss_max = float(valid[OBJECTIVES[1]].max()) if len(valid) else 1.0
vpil_min = float(valid[OBJECTIVES[0]].min()) if len(valid) else 0.0
vpil_max = float(valid[OBJECTIVES[0]].max()) if len(valid) else 1.0

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
    run_prefix = f"[{row['run']}] " if "run" in row.index else ""
    return (
        f"<b>{run_prefix}Sim {int(row['sim_id'])}</b><br>"
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
section_header("Pareto Front", HELP_PARETO)

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
        mode="lines",
        line=dict(width=1.5, color="rgba(99,110,250,0.3)", dash="dot"),
        hoverinfo="skip",
        showlegend=False,
    ))

    if run_choice == "Both":
        trace_groups = [(r, filtered_pareto[filtered_pareto["run"] == r]) for r in RUN_FILES]
    else:
        trace_groups = [(run_choice, filtered_pareto)]

    cost_min = float(filtered_pareto[COST_COL].min())
    cost_max = float(filtered_pareto[COST_COL].max())
    for i, (run_name, group) in enumerate(trace_groups):
        if group.empty:
            continue
        fig_pareto.add_trace(go.Scatter(
            x=group[OBJECTIVES[0]],
            y=group[OBJECTIVES[1]],
            mode="markers",
            marker=dict(
                size=11,
                symbol=RUN_SYMBOLS.get(run_name, "circle"),
                color=group[COST_COL],
                colorscale="Viridis",
                cmin=cost_min,
                cmax=cost_max,
                colorbar=dict(title="Cost") if i == 0 else None,
                showscale=(i == 0),
                line=dict(width=1, color="white"),
            ),
            text=[_hover_text(r) for _, r in group.iterrows()],
            hoverinfo="text",
            name=f"Pareto – {run_name}" if run_choice == "Both" else "Pareto Front",
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
    section_header("Knee Point — Best Trade-off", HELP_KNEE)
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
#  Tabs: Progress / Parameters / Table
# ---------------------------------------------------------------------------
tab_progress, tab_params, tab_table = st.tabs(
    ["Optimization Progress", "Parameter Analysis", "Results Table"]
)

# ----- Tab 1: Optimization Progress -----
with tab_progress:
    section_header("Cost vs. Simulation ID", HELP_COST)

    sorted_valid = valid.sort_values("sim_id")
    if len(sorted_valid) == 0:
        st.info("No valid simulations for this view.")
    else:
        sim_id_min = int(sorted_valid["sim_id"].min())
        sim_id_max = int(sorted_valid["sim_id"].max())
        cost_val_min = float(sorted_valid[COST_COL].min())
        cost_val_max = float(sorted_valid[COST_COL].max())

        cc1, cc2 = st.columns(2)
        sim_id_range = cc1.slider(
            "Sim ID range",
            min_value=sim_id_min,
            max_value=sim_id_max,
            value=(sim_id_min, sim_id_max),
            step=1,
            key="cost_sim_range",
        )
        cost_range = cc2.slider(
            "Cost range",
            min_value=cost_val_min,
            max_value=cost_val_max,
            value=(cost_val_min, cost_val_max),
            step=0.01,
            key="cost_val_range",
        )

        cost_mask = (
            (sorted_valid["sim_id"] >= sim_id_range[0])
            & (sorted_valid["sim_id"] <= sim_id_range[1])
            & (sorted_valid[COST_COL] >= cost_range[0])
            & (sorted_valid[COST_COL] <= cost_range[1])
        )
        cost_filtered = sorted_valid[cost_mask]

        fig_cost = go.Figure()
        cost_groups = (
            [(r, cost_filtered[cost_filtered["run"] == r]) for r in RUN_FILES]
            if run_choice == "Both"
            else [(run_choice, cost_filtered)]
        )
        for run_name, group in cost_groups:
            if group.empty:
                continue
            color = RUN_COLORS.get(run_name, "rgb(99,110,250)")
            suffix = f" – {run_name}" if run_choice == "Both" else ""
            fig_cost.add_trace(go.Scatter(
                x=group["sim_id"],
                y=group[COST_COL],
                mode="markers",
                marker=dict(size=5, color=color, opacity=0.5,
                            symbol=RUN_SYMBOLS.get(run_name, "circle")),
                name=f"Cost per sim{suffix}",
                hovertemplate=f"[{run_name}] Sim %{{x:.0f}}<br>Cost = %{{y:.4f}}<extra></extra>",
            ))
            running_best = group[COST_COL].expanding().min()
            fig_cost.add_trace(go.Scatter(
                x=group["sim_id"],
                y=running_best,
                mode="lines",
                line=dict(width=2.5, color=color),
                name=f"Running best{suffix}",
                hovertemplate=f"[{run_name}] Sim %{{x:.0f}}<br>Best so far = %{{y:.4f}}<extra></extra>",
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

    section_header("LHS vs. BO", HELP_LHS_BO)
    if len(valid) == 0:
        st.info("No data for LHS vs BO comparison.")
    else:
        lhs_n = int((valid["sim_id"] <= lhs_count).sum())
        bo_n = int((valid["sim_id"] > lhs_count).sum())
        pareto_total = int(valid["is_pareto"].sum())
        pareto_bo = int((valid["is_pareto"] & (valid["sim_id"] > lhs_count)).sum())
        pct_pareto_bo = (100.0 * pareto_bo / pareto_total) if pareto_total else 0.0
        med_lhs = valid.loc[valid["sim_id"] <= lhs_count, COST_COL].median()
        med_bo = valid.loc[valid["sim_id"] > lhs_count, COST_COL].median()
        if pd.notna(med_lhs) and pd.notna(med_bo) and med_lhs > 0:
            bo_vs_lhs = med_bo / med_lhs
        else:
            bo_vs_lhs = float("nan")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LHS sims (in view)", lhs_n)
        m2.metric("BO sims (in view)", bo_n)
        m3.metric("% Pareto from BO", f"{pct_pareto_bo:.1f}%")
        m4.metric("Median cost BO / LHS", f"{bo_vs_lhs:.3f}" if np.isfinite(bo_vs_lhs) else "—")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.plotly_chart(
                build_phase_boxplot(valid, COST_COL, "Cost", lhs_count),
                width="stretch",
            )
        with r1c2:
            st.plotly_chart(
                build_phase_boxplot(valid, OBJECTIVES[0], "V_π·L (V·mm)", lhs_count),
                width="stretch",
            )
        st.plotly_chart(
            build_phase_boxplot(valid, OBJECTIVES[1], "Loss (dB/cm)", lhs_count),
            width="stretch",
        )

# ----- Tab 2: Parameter Analysis -----
with tab_params:
    section_header("Parallel Coordinates", HELP_PARALLEL)
    if len(valid) == 0:
        st.info("No valid simulations for parallel coordinates.")
    else:
        pc_pareto_only = st.checkbox("Show Pareto only", value=False, key="pc_pareto_only")
        pc_df_source = valid[valid["is_pareto"]] if pc_pareto_only else valid
        color_by = st.selectbox(
            "Color by",
            ["Cost", "V_π·L (V·mm)", "Loss (dB/cm)"],
            key="pc_color",
        )
        color_map = {
            "Cost": COST_COL,
            "V_π·L (V·mm)": OBJECTIVES[0],
            "Loss (dB/cm)": OBJECTIVES[1],
        }
        color_col = color_map[color_by]
        pc_df = pd.DataFrame({
            "w_r (nm)": pc_df_source["w_r"].to_numpy() * 1e9,
            "h_si (nm)": pc_df_source["h_si"].to_numpy() * 1e9,
            "log10 Doping": np.log10(pc_df_source["doping"].to_numpy()),
            "S (nm)": pc_df_source["S"].to_numpy() * 1e9,
            "λ (nm)": pc_df_source["lambda"].to_numpy() * 1e9,
            "Length (mm)": pc_df_source["length"].to_numpy() * 1e3,
            "V_π·L (V·mm)": pc_df_source[OBJECTIVES[0]].to_numpy(),
            "Loss (dB/cm)": pc_df_source[OBJECTIVES[1]].to_numpy(),
            color_by: pc_df_source[color_col].to_numpy(),
        })
        dims = [c for c in pc_df.columns if c != color_by]
        fig_pc = px.parallel_coordinates(
            pc_df,
            dimensions=dims + [color_by],
            color=color_by,
            color_continuous_scale="Viridis",
        )
        fig_pc.update_layout(
            template="plotly_white",
            height=480,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_pc, width="stretch")

    section_header("Parameter–Target Correlations", HELP_HEATMAP)
    if len(valid) < 2:
        st.info("Need at least 2 valid points for correlations.")
    else:
        target_cols = [OBJECTIVES[0], OBJECTIVES[1], COST_COL]
        corr = valid[PARAM_COLS + target_cols].corr(numeric_only=True).loc[PARAM_COLS, target_cols]
        x_labels = [target_axis_label(c) for c in target_cols]
        y_labels = [param_axis_label(p) for p in PARAM_COLS]
        z = corr.values.astype(float)
        text = np.round(z, 2).astype(str)
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                colorscale="RdBu_r",
                zmid=0,
                text=text,
                texttemplate="%{text}",
                hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>",
            )
        )
        fig_hm.update_layout(
            template="plotly_white",
            height=340,
            margin=dict(l=100, r=40, t=40, b=80),
            xaxis_title="Target",
            yaxis_title="Parameter",
        )
        st.plotly_chart(fig_hm, width="stretch")

    section_header("Parameter Distributions: Pareto vs Other", HELP_DIST)
    if len(valid) == 0:
        st.info("No data for distributions.")
    else:
        pareto_sub = valid[valid["is_pareto"]]
        non_sub = valid[~valid["is_pareto"]]
        sub_titles = [param_axis_label(p) for p in PARAM_COLS]
        fig_dist = make_subplots(rows=2, cols=3, subplot_titles=sub_titles)
        for i, param in enumerate(PARAM_COLS):
            row = i // 3 + 1
            col = i % 3 + 1
            if param == "doping":
                x_p = pareto_sub[param].to_numpy() if len(pareto_sub) else np.array([])
                x_n = non_sub[param].to_numpy() if len(non_sub) else np.array([])
            else:
                scales = {
                    "w_r": 1e9,
                    "h_si": 1e9,
                    "S": 1e9,
                    "lambda": 1e9,
                    "length": 1e3,
                }
                sc = scales[param]
                x_p = pareto_sub[param].to_numpy() * sc if len(pareto_sub) else np.array([])
                x_n = non_sub[param].to_numpy() * sc if len(non_sub) else np.array([])
            fig_dist.add_trace(
                go.Histogram(
                    x=x_n,
                    name="Other",
                    marker_color="rgba(150,150,150,0.55)",
                    showlegend=(i == 0),
                    legendgroup="other",
                ),
                row=row,
                col=col,
            )
            fig_dist.add_trace(
                go.Histogram(
                    x=x_p,
                    name="Pareto",
                    marker_color="rgb(99,110,250)",
                    opacity=0.75,
                    showlegend=(i == 0),
                    legendgroup="pareto",
                ),
                row=row,
                col=col,
            )
            if param == "doping":
                fig_dist.update_xaxes(type="log", row=row, col=col)
        fig_dist.update_layout(
            template="plotly_white",
            height=520,
            barmode="overlay",
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_dist, width="stretch")

# ----- Tab 3: Results Table -----
with tab_table:
    section_header("Results", HELP_TABLE)
    table_view = st.radio(
        "Table view",
        ["Pareto-optimal only", "All valid (filtered)", "All results (raw)"],
        horizontal=True,
        key="table_view_mode",
    )

    if table_view == "Pareto-optimal only":
        table_data = filtered_pareto
        st.caption(f"Pareto-Optimal Designs ({len(table_data)})")
    elif table_view == "All valid (filtered)":
        table_data = valid
        st.caption(f"All Valid Sims ({len(table_data)})")
    else:
        table_data = raw_df.sort_values("sim_id")
        st.caption(f"All Results ({len(table_data)})")

    display_df = to_display_units(table_data, cost_col=COST_COL)
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        height=min(40 * len(display_df) + 50, 600),
    )

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, "pareto_results.csv", "text/csv")
