# mt5_dashboard_mt5_autorun.py
# Run: python mt5_dashboard_mt5_autorun.py
# Auto-runs Streamlit on fixed local port (headless, no auto browser)

from __future__ import annotations

import os
import sys
import subprocess
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import MetaTrader5 as mt5


# -------------------- Launcher --------------------
def _should_autorun_streamlit() -> bool:
    return os.environ.get("ST_AUTORUN_LAUNCHED") != "1" and (
        os.environ.get("STREAMLIT_SERVER_RUNNING") is None
    )


if __name__ == "__main__" and _should_autorun_streamlit():
    os.environ["ST_AUTORUN_LAUNCHED"] = "1"
    PORT = os.environ.get("ST_PORT", "8502")
    cmd = [
        sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__),
        "--server.address", "127.0.0.1",
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    print(f"\n🚀 Streamlit: http://127.0.0.1:{PORT}\n")
    subprocess.run(cmd)
    raise SystemExit


# -------------------- UI / CSS --------------------
st.set_page_config(page_title="MT5 Live Report Dashboard", page_icon="📈", layout="wide")

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }
.small { color: rgba(255,255,255,0.65); font-size: 0.92rem; }

.card {
  padding: 16px 16px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  background: linear-gradient(135deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01));
}
.card h3 { margin: 0 0 6px 0; font-size: 0.95rem; font-weight: 800; color: rgba(255,255,255,0.70); }
.big { font-size: 1.70rem; font-weight: 950; margin: 2px 0 6px 0; letter-spacing: -0.02em; }
.sub { font-size: 0.86rem; color: rgba(255,255,255,0.62); line-height: 1.45; }
.good { color: #2EE59D; }
.bad  { color: #FF5C7A; }
.neu  { color: rgba(255,255,255,0.92); }

.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.80rem;
  font-weight: 900;
  margin-left: 8px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
}
.badge.good { border-color: rgba(46,229,157,0.35); background: rgba(46,229,157,0.08); }
.badge.bad  { border-color: rgba(255,92,122,0.35); background: rgba(255,92,122,0.08); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -------------------- Helpers --------------------
def format_money(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.2f}"


def kpi_cls(net: float) -> str:
    if net > 0:
        return "good"
    if net < 0:
        return "bad"
    return "neu"


def pnl_summary(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"gp": 0.0, "gl": 0.0, "net": 0.0, "trades": 0, "wins": 0, "losses": 0}

    p = df["profit"].astype(float)
    gp = float(p[p > 0].sum())
    gl = float(p[p < 0].sum())  # negative
    net = float(p.sum())
    return {
        "gp": gp,
        "gl": gl,
        "net": net,
        "trades": int(len(p)),
        "wins": int((p > 0).sum()),
        "losses": int((p < 0).sum()),
    }


def render_card(title: str, badge_text: str, s: dict):
    cls = kpi_cls(s["net"])
    bcls = "good" if s["net"] > 0 else ("bad" if s["net"] < 0 else "")
    st.markdown(
        f"""
        <div class="card">
          <h3>{title} <span class="badge {bcls}">{badge_text}</span></h3>
          <div class="big {cls}">{format_money(s["net"])}</div>
          <div class="sub">
            <span class="good">Profit: {format_money(s["gp"])}</span>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <span class="bad">Loss: {format_money(s["gl"])}</span><br/>
            Trades: <b>{s["trades"]}</b> &nbsp; Wins: <b>{s["wins"]}</b> &nbsp; Losses: <b>{s["losses"]}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def period_table(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    freq:
      'D'      -> daily
      'W-MON'  -> weekly (week starts Monday)
      'M'      -> monthly
    """
    x = df.copy()
    x["close_time"] = pd.to_datetime(x["close_time"], errors="coerce")
    x = x.dropna(subset=["close_time"]).set_index("close_time").sort_index()

    if x.empty:
        return pd.DataFrame(columns=["period", "gross_profit", "gross_loss", "net", "trades", "wins", "losses"])

    def _agg(g: pd.DataFrame) -> pd.Series:
        p = g["profit"].astype(float)
        return pd.Series({
            "gross_profit": float(p[p > 0].sum()),
            "gross_loss": float(p[p < 0].sum()),  # negative
            "net": float(p.sum()),
            "trades": int(len(p)),
            "wins": int((p > 0).sum()),
            "losses": int((p < 0).sum()),
        })

    out = x.resample(freq).apply(_agg).reset_index()
    out = out.rename(columns={"close_time": "period"})
    return out


def last_period_summary(tbl: pd.DataFrame) -> tuple[str, dict]:
    """
    Returns (badge, summary_dict) for the LAST period row in the table.
    """
    if tbl is None or tbl.empty:
        return "-", {"gp": 0.0, "gl": 0.0, "net": 0.0, "trades": 0, "wins": 0, "losses": 0}

    r = tbl.sort_values("period").iloc[-1]
    badge = str(pd.to_datetime(r["period"]).date()) if isinstance(r["period"], (pd.Timestamp, datetime)) else str(r["period"])
    s = {
        "gp": float(r["gross_profit"]),
        "gl": float(r["gross_loss"]),
        "net": float(r["net"]),
        "trades": int(r["trades"]),
        "wins": int(r["wins"]),
        "losses": int(r["losses"]),
    }
    return badge, s


def build_equity_from_trades(trades: pd.DataFrame, initial_equity: float) -> pd.DataFrame:
    eq = trades[["close_time", "profit"]].dropna(subset=["close_time"]).copy()
    eq = eq.sort_values("close_time")
    eq["equity"] = float(initial_equity) + eq["profit"].cumsum()
    eq = eq.rename(columns={"close_time": "time"})
    return eq


def compute_drawdown(equity: pd.Series):
    peak = equity.cummax()
    dd = equity - peak
    dd_pct = dd / peak.replace(0, np.nan)
    max_dd_abs = float(dd.min()) if len(dd) else 0.0
    max_dd_pct = float(dd_pct.min()) if len(dd_pct) else 0.0
    return dd, abs(max_dd_abs), abs(max_dd_pct)


# -------------------- MT5 Connect / Fetch --------------------
def mt5_connect_auto(path: str | None = None) -> tuple[bool, str]:
    ok = mt5.initialize(path.strip()) if path and path.strip() else mt5.initialize()
    if not ok:
        return False, f"MT5 initialize failed: {mt5.last_error()}"

    ai = mt5.account_info()
    if ai is None:
        return False, f"MT5 account_info() failed: {mt5.last_error()}"

    return True, f"Connected ✅ | Account: {ai.login} | Server: {ai.server}"


def fetch_deals(from_dt: datetime, to_dt: datetime, include_costs: bool = True) -> pd.DataFrame:
    deals = mt5.history_deals_get(from_dt, to_dt)
    if deals is None:
        raise RuntimeError(f"history_deals_get returned None: {mt5.last_error()}")

    if len(deals) == 0:
        return pd.DataFrame(columns=["symbol","close_time","profit","type","volume","price","comment","ticket","position_id"])

    first = deals[0]
    if hasattr(first, "_asdict"):
        df = pd.DataFrame([d._asdict() for d in deals])
    else:
        df = pd.DataFrame(list(deals))

    # numeric columns -> map schema (your broker returned 0..17 earlier)
    if all(isinstance(c, int) for c in df.columns):
        schema18 = [
            "ticket","order","time","time_msc","type","entry","magic","position_id","reason",
            "volume","price","commission","swap","profit","fee","symbol","comment","external_id"
        ]
        if len(df.columns) == 18:
            df.columns = schema18
        else:
            raise RuntimeError(f"Unexpected deals tuple width={len(df.columns)}. cols={list(df.columns)}")

    # time
    if "time" in df.columns:
        df["close_time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
    elif "time_msc" in df.columns:
        df["close_time"] = pd.to_datetime(df["time_msc"], unit="ms", errors="coerce")
    else:
        raise RuntimeError(f"Deal time column not found after mapping. cols={list(df.columns)}")

    for col in ["profit", "commission", "swap", "fee", "volume", "price", "ticket", "position_id", "type"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # net profit
    if include_costs:
        profit_net = df.get("profit", 0).fillna(0)
        if "swap" in df.columns:
            profit_net = profit_net + df["swap"].fillna(0)
        if "commission" in df.columns:
            profit_net = profit_net + df["commission"].fillna(0)
        if "fee" in df.columns:
            profit_net = profit_net + df["fee"].fillna(0)
        df["profit_net"] = profit_net
    else:
        df["profit_net"] = df.get("profit", 0).fillna(0)

    # only BUY/SELL deals
    type_map = {mt5.DEAL_TYPE_BUY: "BUY", mt5.DEAL_TYPE_SELL: "SELL"}
    df["type_txt"] = df.get("type", np.nan).map(type_map)

    if "type" in df.columns:
        df = df[df["type"].isin([mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL])].copy()

    out = pd.DataFrame({
        "symbol": df.get("symbol", "").astype(str),
        "close_time": df["close_time"],
        "profit": df["profit_net"].astype(float),
        "type": df.get("type_txt", "").astype(str),
        "volume": df.get("volume", np.nan),
        "price": df.get("price", np.nan),
        "comment": df.get("comment", "").astype(str),
        "ticket": df.get("ticket", np.nan),
        "position_id": df.get("position_id", np.nan),
    })

    out = out.dropna(subset=["close_time", "profit"]).sort_values("close_time").reset_index(drop=True)
    return out


# -------------------- State --------------------
if "mt5_connected" not in st.session_state:
    st.session_state.mt5_connected = False
    st.session_state.mt5_status = "Not connected"

if "cache_key" not in st.session_state:
    st.session_state.cache_key = None
if "cache_deals" not in st.session_state:
    st.session_state.cache_deals = pd.DataFrame()

# Auto connect
if not st.session_state.mt5_connected:
    ok, msg = mt5_connect_auto(path=None)
    st.session_state.mt5_connected = ok
    st.session_state.mt5_status = msg


# -------------------- Main UI --------------------
st.title("📈 MT5 Live Report Dashboard")
st.markdown('<div class="small">✅ تاریخ را تغییر بده؛ گزارش و لیست معاملات باید دقیقاً همان لحظه فیلتر شود.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🗓️ Date Range")
    default_to = date.today()
    default_from = default_to - timedelta(days=30)
    d_from, d_to = st.date_input("From / To", value=(default_from, default_to))
    st.divider()

    st.header("⚙️ Settings")
    initial_equity = st.number_input("Initial equity (for equity curve)", min_value=0.0, value=10000.0, step=100.0)
    include_costs = st.toggle("Include commission/swap in PnL", value=True)
    st.divider()

    refresh_btn = st.button("🔄 Force refresh", use_container_width=True)

st.info(st.session_state.mt5_status)
if not st.session_state.mt5_connected:
    st.warning("MT5 وصل نشد. MT5 را باز کن و لاگین باش.")
    st.stop()

# Convert date range
from_dt = datetime.combine(d_from, datetime.min.time())
to_dt = datetime.combine(d_to, datetime.max.time())

# ✅ Correct caching key: if date/include_costs changes => refetch automatically
key = (str(from_dt), str(to_dt), bool(include_costs))
need_fetch = refresh_btn or (st.session_state.cache_key != key)

if need_fetch:
    try:
        deals_df = fetch_deals(from_dt, to_dt, include_costs=include_costs)
        st.session_state.cache_key = key
        st.session_state.cache_deals = deals_df
    except Exception as e:
        st.error(f"خطا در گرفتن history از MT5: {e}")
        st.stop()
else:
    deals_df = st.session_state.cache_deals.copy()

if deals_df.empty:
    st.warning("در این بازه هیچ Deal (BUY/SELL) نیست. تاریخ را تغییر بده.")
    st.stop()

# Symbol filter (this now truly filters the tables too)
symbols = sorted(deals_df["symbol"].dropna().unique().tolist())
sym_sel = st.multiselect("Symbol filter", options=symbols, default=symbols)

flt = deals_df[deals_df["symbol"].isin(sym_sel)].copy()
if flt.empty:
    st.warning("بعد از فیلتر Symbol دیتایی باقی نموند.")
    st.stop()

# -------------------- Summary Boxes (Correct) --------------------
st.write("")
st.subheader("💎 Summary Boxes (Based on Selected Date Range)")

# Build period tables inside selected range
daily_tbl = period_table(flt, "D")
weekly_tbl = period_table(flt, "W-MON")
monthly_tbl = period_table(flt, "M")

# Last day/week/month inside range
daily_badge, daily_sum = last_period_summary(daily_tbl)
weekly_badge, weekly_sum = last_period_summary(weekly_tbl)
monthly_badge, monthly_sum = last_period_summary(monthly_tbl)

range_sum = pnl_summary(flt)

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_card("Daily (Last day in range)", daily_badge, daily_sum)
with c2:
    render_card("Weekly (Last week in range)", weekly_badge, weekly_sum)
with c3:
    render_card("Monthly (Last month in range)", monthly_badge, monthly_sum)
with c4:
    render_card("Selected Range (Total)", f"{d_from} → {d_to}", range_sum)

# Optional: show the period tables (so you can verify)
with st.expander("🔎 Verify PnL tables (Daily/Weekly/Monthly)", expanded=False):
    t1, t2, t3 = st.tabs(["Daily", "Weekly", "Monthly"])
    with t1:
        st.dataframe(daily_tbl, use_container_width=True, height=260)
    with t2:
        st.dataframe(weekly_tbl, use_container_width=True, height=260)
    with t3:
        st.dataframe(monthly_tbl, use_container_width=True, height=260)

# -------------------- Equity & Drawdown --------------------
st.write("")
st.subheader("📈 Equity & Drawdown")

eq_f = build_equity_from_trades(flt, initial_equity=initial_equity)
dd, max_dd_abs, max_dd_pct = compute_drawdown(eq_f["equity"])

cc1, cc2 = st.columns([2, 1])
with cc1:
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=eq_f["time"], y=eq_f["equity"], mode="lines", name="Equity"))
    fig_eq.update_layout(title="Equity Curve (from deals)", height=360,
                         margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    st.plotly_chart(fig_eq, use_container_width=True)

with cc2:
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=eq_f["time"], y=dd, mode="lines", name="Drawdown"))
    fig_dd.update_layout(title=f"Drawdown (Max: {format_money(max_dd_abs)})", height=360,
                         margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    st.plotly_chart(fig_dd, use_container_width=True)

# -------------------- Trades list (correctly filtered by date+symbol) --------------------
st.write("")
st.subheader("🧾 Trades (Filtered)")

st.dataframe(
    flt.sort_values("close_time", ascending=False)[["close_time","symbol","type","volume","price","profit","comment","ticket"]],
    use_container_width=True,
    height=420
)

# Best / Worst
st.write("")
b1, b2 = st.columns(2)
with b1:
    st.subheader("🏆 Best deals")
    top = flt.sort_values("profit", ascending=False).head(20).copy()
    st.dataframe(top[["close_time","symbol","type","volume","price","profit","comment","ticket"]],
                 use_container_width=True, height=320)

with b2:
    st.subheader("💥 Worst deals")
    bot = flt.sort_values("profit", ascending=True).head(20).copy()
    st.dataframe(bot[["close_time","symbol","type","volume","price","profit","comment","ticket"]],
                 use_container_width=True, height=320)

# Export
st.write("")
st.subheader("⬇️ Export")
out = flt.copy()
out["close_time"] = out["close_time"].astype(str)
st.download_button(
    "Download filtered deals CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="mt5_deals_filtered.csv",
    mime="text/csv",
)
