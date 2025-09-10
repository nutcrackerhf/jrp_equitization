# app.py
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Lump Sum vs DCA — Shiller Empirical Explorer",
    layout="wide",
)

# -----------------------------
# Utilities & Caching
# -----------------------------
DEFAULT_XLSX = "shiller_base_data.xlsx"
COLUMNS = {
    "date": "Date",
    "eq_index": "SP500 Real Total Return Index",       # real TR index level
    "bond_index": "10yBond Real Total Return Index",   # real TR index level
    "cape": "Cyclically Adjusted PE Ratio",
}

@st.cache_data
def load_shiller_from_excel(path: str, cols: dict, lag_cape_months: int) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Basic validations
    for k, v in cols.items():
        if v not in df.columns:
            raise ValueError(f"Expected column '{v}' not found for '{k}' in {path}")
    df = df.sort_values(cols["date"]).reset_index(drop=True)
    # Convert to monthly returns (index levels → pct_change)
    df["eq_ret"]   = df[cols["eq_index"]].astype(float).pct_change()
    df["bond_ret"] = df[cols["bond_index"]].astype(float).pct_change()
    # Implementability: lag CAPE by chosen months
    df["cape_lag"] = df[cols["cape"]].shift(lag_cape_months)
    # Tidy
    df = df.dropna(subset=["eq_ret", "bond_ret", "cape_lag"]).reset_index(drop=True)
    df[cols["date"]] = pd.to_datetime(df[cols["date"]])
    df = df.rename(columns={cols["date"]: "Date"})
    return df

def monthly_cash_return(y_annual: float) -> float:
    return (1.0 + y_annual) ** (1/12) - 1.0

def first_year_mdd(path_like) -> float:
    """Max drawdown within first 12 months of a wealth path."""
    if len(path_like) == 0:
        return np.nan
    n = min(12, len(path_like))
    first = np.array(path_like[:n], dtype=float)
    peak = np.maximum.accumulate(first)
    dd = first / np.where(peak == 0, np.nan, peak) - 1.0
    return float(np.nanmin(dd))

def simulate_path(eq, bd, H, weights, cash_m, K=0, partial_x=1.0):
    """
    Generic simulator: rebalances invested sleeve to 'weights' monthly.
    - eq, bd: arrays of monthly returns (len >= H)
    - H: horizon months
    - weights: (w_eq, w_bond)
    - cash_m: monthly cash rate
    - K: DCA months (0 = lump everything immediately; >0 spreads remaining cash evenly over K months)
    - partial_x: fraction invested at t=0 (for partial LS + DCA)
    """
    w_eq, w_bd = weights
    wealth_cash = 1.0 - partial_x
    invested = partial_x
    eq_h = invested * w_eq
    bd_h = invested * w_bd
    installment = 0 if K <= 0 else max(0.0, (1.0 - partial_x) / K)

    wealth_path = []
    for m in range(H):
        # If DCA is active, invest evenly from cash for first K months
        if K > 0 and m < K and installment > 0:
            invest_now = min(wealth_cash, installment)
            wealth_cash -= invest_now
            invested += invest_now
            eq_h += invest_now * w_eq
            bd_h += invest_now * w_bd

        # Apply market returns
        eq_h *= (1.0 + eq[m])
        bd_h *= (1.0 + bd[m])
        wealth_cash *= (1.0 + cash_m)

        # Total
        total = eq_h + bd_h + wealth_cash

        # Rebalance invested sleeve (keep cash separate)
        inv = eq_h + bd_h
        if inv > 0:
            eq_h, bd_h = inv * w_eq, inv * w_bd

        wealth_path.append(total)

    wealth_path = np.array(wealth_path, dtype=float)
    return {
        "wealth_path": wealth_path,
        "end_wealth": float(wealth_path[-1]),
        "first12_mdd": first_year_mdd(wealth_path),
        "underwater12": bool(wealth_path[min(11, H-1)] < 1.0),
    }

def simulate_ls_delay(df, start_idx, H, weights, cash_m, high, low, max_wait_months, force_end=True):
    """
    LS-Delay rule:
      - If start CAPE_lag <= high: invest immediately.
      - Else wait for CAPE_lag <= low (natural trigger).
      - Else force-invest at min(max_wait-1, H-1) if max_wait is set and >0.
      - Else (only when no max_wait set) force-invest at horizon end if force_end=True.
    """
    w_eq, w_bd = weights
    wealth_cash = 1.0
    eq_h = 0.0
    bd_h = 0.0

    cape0 = df.loc[start_idx, "cape_lag"]
    trigger_month = None
    trigger_reason = None

    if cape0 <= high:
        trigger_month = 0
        trigger_reason = "immediate"
    else:
        # Natural trigger
        for m in range(H):
            if df.loc[start_idx + m, "cape_lag"] <= low:
                trigger_month = m
                trigger_reason = "natural"
                break
        # Max-wait
        if trigger_month is None and max_wait_months and max_wait_months > 0:
            tm = min(max_wait_months - 1, H - 1)
            trigger_month = tm
            trigger_reason = "max_wait"
        # Horizon end (only if no max-wait)
        if trigger_month is None and force_end and (not max_wait_months or max_wait_months <= 0):
            trigger_month = H - 1
            trigger_reason = "horizon_end"

    wealth_path = []
    for m in range(H):
        # Invest at trigger month if not yet invested
        if (trigger_month is not None) and (m == trigger_month) and (eq_h + bd_h) == 0.0:
            invest_amt = wealth_cash
            wealth_cash = 0.0
            eq_h = invest_amt * w_eq
            bd_h = invest_amt * w_bd

        # Apply returns
        eq_h *= (1.0 + df["eq_ret"].values[start_idx + m])
        bd_h *= (1.0 + df["bond_ret"].values[start_idx + m])
        wealth_cash *= (1.0 + cash_m)

        total = eq_h + bd_h + wealth_cash

        # Rebalance invested sleeve
        inv = eq_h + bd_h
        if inv > 0:
            eq_h, bd_h = inv * w_eq, inv * w_bd

        wealth_path.append(total)

    wp = np.array(wealth_path, dtype=float)
    return {
        "wealth_path": wp,
        "end_wealth": float(wp[-1]),
        "first12_mdd": first_year_mdd(wp),
        "underwater12": bool(wp[min(11, H-1)] < 1.0),
        "trigger_month": int(trigger_month) if trigger_month is not None else None,
        "trigger_reason": trigger_reason,
        "wait_months": int(trigger_month) if trigger_month is not None else H,
    }

@st.cache_data(show_spinner=False)
def enumerate_outcomes(
    df: pd.DataFrame,
    weights=(0.7, 0.3),
    horizons=(12, 24, 36, 60),
    dca_months=(3, 6, 9, 12, 18, 24),
    partial_fracs=(0.0, 0.25, 0.5, 0.75),
    cash_real_y=0.0,
    ls_delay_cfg=None,
):
    cash_m = monthly_cash_return(cash_real_y)
    maxH = max(horizons)
    rows = []

    for start in range(len(df) - maxH):
        eq = df["eq_ret"].values[start:start+maxH]
        bd = df["bond_ret"].values[start:start+maxH]
        meta = {
            "start_idx": start,
            "start_date": df["Date"].iloc[start],
            "cape_lag": df["cape_lag"].iloc[start],
        }

        for H in horizons:
            # LS
            res_ls = simulate_path(eq, bd, H, weights, cash_m, K=0, partial_x=1.0)
            rows.append({**meta, "horizon_m": H, "strategy": "LS", "K": 0, "partial_x": 1.0, **res_ls})

            # DCA (pure DCA from cash)
            for K in dca_months:
                res_dca = simulate_path(eq, bd, H, weights, cash_m, K=K, partial_x=0.0)
                rows.append({**meta, "horizon_m": H, "strategy": "DCA", "K": K, "partial_x": 0.0, **res_dca})

            # Partial + DCA
            for x in partial_fracs:
                if 0.0 < x < 1.0:
                    for K in dca_months:
                        res_p = simulate_path(eq, bd, H, weights, cash_m, K=K, partial_x=x)
                        rows.append({**meta, "horizon_m": H, "strategy": "PARTIAL", "K": K, "partial_x": x, **res_p})

            # LS-Delay
            if ls_delay_cfg is not None:
                res_ld = simulate_ls_delay(
                    df, start, H, weights, cash_m,
                    high=ls_delay_cfg["high"], low=ls_delay_cfg["low"],
                    max_wait_months=ls_delay_cfg["max_wait"], force_end=ls_delay_cfg["force_end"]
                )
                rows.append({**meta, "horizon_m": H, "strategy": "LS_DELAY", "K": np.nan, "partial_x": np.nan, **res_ld})

    out = pd.DataFrame(rows)
    out["total_return"] = out["end_wealth"] - 1.0
    out["annualized"] = out.apply(lambda r: r["end_wealth"] ** (12.0 / r["horizon_m"]) - 1.0, axis=1)
    return out

def to_bytes_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# -----------------------------
# Data Ingest (Local file or Upload)
# -----------------------------
st.sidebar.header("Data")
lag_cape_default = 1
uploaded = st.sidebar.file_uploader("Upload Shiller Excel (optional)", type=["xlsx"])

try:
    if uploaded is not None:
        df = load_shiller_from_excel(uploaded, COLUMNS, lag_cape_default)
        st.sidebar.success("Loaded uploaded Excel.")
    elif os.path.exists(DEFAULT_XLSX):
        df = load_shiller_from_excel(DEFAULT_XLSX, COLUMNS, lag_cape_default)
        st.sidebar.info(f"Using bundled file: {DEFAULT_XLSX}")
    else:
        st.error("Please upload 'shiller_base_data.xlsx' (or add it to the repo).")
        st.stop()
except Exception as e:
    st.error(f"Problem loading data: {e}")
    st.stop()

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Portfolio & Simulation")
w_eq = st.sidebar.slider("Equity weight", 0.0, 1.0, 0.70, 0.05)
w_bd = 1.0 - w_eq
cash_real = st.sidebar.number_input("Cash real yield (annual, %)", value=0.00, step=0.25) / 100
horizons = st.sidebar.multiselect("Horizons (months)", [12, 24, 36, 60], default=[12, 60])
K_list = st.sidebar.multiselect("DCA months (K)", [3, 6, 9, 12, 18, 24], default=[6, 12])
partials = st.sidebar.multiselect("Partial lump-sum (x immediately)", [0.25, 0.5, 0.75], default=[0.5])

st.sidebar.header("CAPE Settings")
lag_cape = st.sidebar.number_input("CAPE lag (months)", min_value=0, value=1, step=1)
if lag_cape != lag_cape_default:
    # Reload with new lag if needed (for uploaded or bundled file)
    try:
        if uploaded is not None:
            df = load_shiller_from_excel(uploaded, COLUMNS, lag_cape)
        else:
            df = load_shiller_from_excel(DEFAULT_XLSX, COLUMNS, lag_cape)
    except Exception as e:
        st.error(f"Problem reloading with CAPE lag={lag_cape}: {e}")
        st.stop()

high_thr = st.sidebar.number_input("LS-Delay: High threshold (wait if CAPE >", min_value=0.0, value=25.0, step=1.0)
low_thr  = st.sidebar.number_input("LS-Delay: Invest when CAPE ≤", min_value=0.0, value=20.0, step=1.0)
max_wait = st.sidebar.number_input("LS-Delay: Max wait (months)", min_value=0, value=24, step=1)
force_end = st.sidebar.checkbox("Force invest at horizon end if no max-wait", value=True)

st.sidebar.header("Run")
do_run = st.sidebar.button("Run / Recompute")

# CAPE bucketing choice
st.sidebar.header("CAPE Buckets")
use_deciles = st.sidebar.checkbox("Use deciles (Q1–Q10)", value=True)
if use_deciles:
    df["cape_bucket"] = pd.qcut(df["cape_lag"], 10, labels=[f"Q{i}" for i in range(1, 11)])
else:
    bins = [0, 15, 20, 25, 30, 1000]
    labels = ["<15", "15-20", "20-25", "25-30", ">=30"]
    df["cape_cut"] = pd.cut(df["cape_lag"], bins=bins, labels=labels, include_lowest=True)

# -----------------------------
# Simulate
# -----------------------------
if do_run:
    ls_delay_cfg = {"high": high_thr, "low": low_thr, "max_wait": max_wait, "force_end": force_end}
    outcomes = enumerate_outcomes(
        df,
        weights=(w_eq, w_bd),
        horizons=tuple(sorted(set(horizons))),
        dca_months=tuple(sorted(set(K_list))),
        partial_fracs=tuple(sorted(set([0.0] + partials))),
        cash_real_y=cash_real,
        ls_delay_cfg=ls_delay_cfg
    )
    st.success(f"Simulated {len(outcomes):,} strategy start-dates × horizons.")
else:
    st.info("Adjust settings in the sidebar and click **Run / Recompute**.")
    st.stop()

# Ensure a simple CAPE cut exists for grouping (if user chose deciles)
if "cape_cut" not in outcomes.columns:
    bins = [0, 15, 20, 25, 30, 1000]
    labels = ["<15", "15-20", "20-25", "25-30", ">=30"]
    meta = outcomes[["start_idx"]].drop_duplicates().merge(
        df.reset_index().rename(columns={"index": "start_idx"})[["start_idx", "cape_lag"]],
        on="start_idx",
        how="left"
    )
    cut_map = pd.cut(meta["cape_lag"], bins=bins, labels=labels, include_lowest=True)
    outcomes = outcomes.merge(pd.DataFrame({"start_idx": meta["start_idx"], "cape_cut": cut_map}), on="start_idx", how="left")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Win-Rate Heatmap",
    "CAPE vs 5-Year Regret",
    "Median Outcomes (LS vs DCA-K)",
    "LS-Delay (Triggers & Outcomes)",
])

# --- Tab 1: Heatmap ---
with tab1:
    st.subheader("DCA vs LS — 12-Month Win Rate by CAPE Decile and K")
    H_heat = 12
    if use_deciles:
        group = outcomes[(outcomes["horizon_m"] == H_heat) & (outcomes["strategy"].isin(["LS", "DCA"]))]
        ls_w = group[group["strategy"] == "LS"][["start_idx", "end_wealth"]].rename(columns={"end_wealth": "LS"})
        heat = None
        for k in sorted(set(K_list)):
            dca_w = group[(group["strategy"] == "DCA") & (group["K"] == k)][["start_idx", "end_wealth", "cape_bucket"]].rename(
                columns={"end_wealth": "DCA", "cape_bucket": "bucket"}
            )
            m = dca_w.merge(ls_w, on="start_idx")
            m["win"] = (m["DCA"] > m["LS"]).astype(int)
            wr = m.groupby("bucket")["win"].mean()
            if heat is None:
                heat = pd.DataFrame(wr)
                heat.columns = [k]
            else:
                heat[k] = wr
        heat = heat.reindex([f"Q{i}" for i in range(1, 11)])
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="viridis", ax=ax)
        ax.set_title("DCA Win Rate vs LS (H=12m)")
        st.pyplot(fig)
        st.download_button("Download heatmap data (CSV)", data=to_bytes_csv(heat.reset_index()), file_name="winrate_heatmap_data.csv")
    else:
        st.info("Switch to deciles in the sidebar to view the heatmap.")

# --- Tab 2: CAPE vs 5-Year Regret ---
with tab2:
    st.subheader("Starting CAPE vs 5-Year Regret (DCA-K vs LS)")
    H_reg = st.selectbox("Horizon for regret scatter", options=sorted(set(horizons)), index=(sorted(set(horizons)).index(60) if 60 in horizons else 0))
    K_reg = st.selectbox("DCA months (K) for regret", options=sorted(set(K_list)), index=(sorted(set(K_list)).index(12) if 12 in K_list else 0))
    lsH = outcomes[(outcomes["strategy"] == "LS") & (outcomes["horizon_m"] == H_reg)][["start_idx", "end_wealth", "cape_lag"]].rename(columns={"end_wealth": "LS_end"})
    dcaH = outcomes[(outcomes["strategy"] == "DCA") & (outcomes["horizon_m"] == H_reg) & (outcomes["K"] == K_reg)][["start_idx", "end_wealth"]].rename(columns={"end_wealth": "DCA_end"})
    m = lsH.merge(dcaH, on="start_idx", how="inner")
    m["regret"] = m["DCA_end"] - m["LS_end"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(m["cape_lag"], m["regret"], alpha=0.4)
    ax.axhline(0, ls="--", lw=1)
    ax.set_xlabel("Starting CAPE (lagged)")
    ax.set_ylabel("Regret (DCA - LS)")
    ax.set_title(f"Regret at {H_reg} months, K={K_reg}")
    st.pyplot(fig)
    st.download_button("Download scatter data (CSV)", data=to_bytes_csv(m), file_name="cape_vs_regret_scatter_data.csv")

# --- Tab 3: Median Outcomes (LS vs DCA-K) ---
with tab3:
    st.subheader("Median Outcomes by CAPE Cut (LS vs DCA-K)")
    K_med = st.selectbox("DCA months (K) for comparison", options=sorted(set(K_list)), index=(sorted(set(K_list)).index(12) if 12 in K_list else 0))
    H_sel = st.selectbox("Horizon", options=sorted(set(horizons)), index=0)
    sub = outcomes[(outcomes["horizon_m"] == H_sel) & (outcomes["strategy"].isin(["LS", "DCA"])) & ((outcomes["K"] == K_med) | (outcomes["strategy"] == "LS"))]
    ls_tab = sub[sub["strategy"] == "LS"].groupby("cape_cut").agg(med_ret=("total_return", "median"), mdd12=("first12_mdd", "mean"))
    dca_tab = sub[(sub["strategy"] == "DCA") & (sub["K"] == K_med)].groupby("cape_cut").agg(med_ret=("total_return", "median"), mdd12=("first12_mdd", "mean"))
    comp = ls_tab.join(dca_tab, lsuffix="_LS", rsuffix="_DCA")
    comp["ret_diff"] = comp["med_ret_DCA"] - comp["med_ret_LS"]
    comp["mdd12_diff"] = comp["mdd12_DCA"] - comp["mdd12_LS"]
    st.dataframe((comp * 1.0).round(3))
    st.download_button("Download table (CSV)", data=to_bytes_csv(comp.reset_index()), file_name="median_outcomes_LS_vs_DCAK.csv")

# --- Tab 4: LS-Delay ---
with tab4:
    st.subheader("LS-Delay: Trigger Profile and Outcomes vs LS")

    # Trigger profile
    if "strategy" in outcomes.columns and (outcomes["strategy"] == "LS_DELAY").any():
        ld = outcomes[outcomes["strategy"] == "LS_DELAY"].copy()
        trig = ld.groupby(["cape_cut", "horizon_m"]).agg(
            mean_wait_months=("wait_months", "mean"),
            p_immediate=("trigger_reason", lambda s: np.mean(s == "immediate")),
            p_natural=("trigger_reason", lambda s: np.mean(s == "natural")),
            p_max_wait=("trigger_reason", lambda s: np.mean(s == "max_wait")),
            p_horizon_end=("trigger_reason", lambda s: np.mean(s == "horizon_end")),
        ).reset_index().round(3)
        st.markdown("**Trigger profile (share of cases and average wait):**")
        st.dataframe(trig)
        st.download_button("Download trigger table (CSV)", data=to_bytes_csv(trig), file_name="ls_delay_trigger_profile.csv")

        # Outcomes vs LS
        def summarize(df_):
            return df_.groupby(["cape_cut", "horizon_m"]).agg(
                median_return=("end_wealth", lambda x: x.median() - 1),
                mean_return=("end_wealth", lambda x: x.mean() - 1),
                mdd12_mean=("first12_mdd", "mean"),
                underwater12_p=("underwater12", "mean"),
            ).reset_index()

        ls_sum = summarize(outcomes[outcomes["strategy"] == "LS"])
        ld_sum = summarize(ld)
        comp2 = ls_sum.merge(ld_sum, on=["cape_cut", "horizon_m"], suffixes=("_LS", "_Delay"))
        comp2["return_diff"] = comp2["median_return_Delay"] - comp2["median_return_LS"]
        comp2["drawdown_diff"] = comp2["mdd12_mean_Delay"] - comp2["mdd12_mean_LS"]
        comp2 = comp2.round(3)

        st.markdown("**Performance outcomes (returns & drawdowns): LS vs LS-Delay**")
        st.dataframe(comp2)
        st.download_button("Download outcomes table (CSV)", data=to_bytes_csv(comp2), file_name="ls_delay_vs_ls_outcomes.csv")
    else:
        st.info("No LS-Delay rows yet — enable LS-Delay in the sidebar and click Run.")

st.caption("Tip: Use the sidebar to change K, horizon, CAPE thresholds, max-wait, cash yield, and weights. Download any displayed table for your memo/white paper.")
