#!/usr/bin/env python3
"""
Polymarket Term Structure Pairs — Streamlit Dashboard
=====================================================
Reads trade history and pair state from bundled SQLite DB.
Also fetches live prices from Polymarket API for current market view.

Hosted on Streamlit Cloud: the DB ships as a snapshot with the repo.
The local monitor (polymarket_monitor.py) keeps it updated — push
new DB snapshots to refresh the hosted dashboard.

Usage:
  Local:            streamlit run streamlit_app.py
  Streamlit Cloud:  auto-deployed from GitHub
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import requests
import json
import os
import shutil
import time
from datetime import datetime, timedelta, timezone

# ============================================================
# CONFIG
# ============================================================

REPO_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polymarket_monitor.db")
# Streamlit Cloud: filesystem is read-only except /tmp
RUNTIME_DB = "/tmp/polymarket_monitor.db"
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

WATCHED_SLUGS = [
    'us-forces-enter-iran-by', 'will-trump-visit-china-by',
    'netanyahu-out-before-2027', 'starmer-out-in-2025',
    'will-metamask-launch-a-token-in-2025',
    'will-russia-capture-kostyantynivka-by',
    'microstrategy-sell-any-bitcoin-in-2025',
    'will-base-launch-a-token-in-2025-341',
    'scotus-accepts-sports-event-contract-case-by-july-31-2026',
    'will-russia-capture-lyman-in-2025',
    'when-will-bitcoin-hit-150k',
    'gpt-6-released-by',
    'nato-x-russia-military-clash-in-2025',
]

FEE_PARAMS = {
    'crypto':    {'fee_rate': 0.072, 'exponent': 1},
    'sports':    {'fee_rate': 0.03,  'exponent': 1},
    'finance':   {'fee_rate': 0.04,  'exponent': 1},
    'politics':  {'fee_rate': 0.04,  'exponent': 1},
    'economics': {'fee_rate': 0.03,  'exponent': 0.5},
    'culture':   {'fee_rate': 0.05,  'exponent': 1},
    'weather':   {'fee_rate': 0.025, 'exponent': 0.5},
}

st.set_page_config(
    page_title="Polymarket Pairs Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# DB SETUP
# ============================================================

def get_db_path():
    """Copy bundled DB to /tmp if needed (Streamlit Cloud is read-only)."""
    if os.path.exists(REPO_DB):
        if not os.path.exists(RUNTIME_DB) or \
           os.path.getmtime(REPO_DB) > os.path.getmtime(RUNTIME_DB):
            shutil.copy2(REPO_DB, RUNTIME_DB)
        return RUNTIME_DB
    return REPO_DB


@st.cache_resource
def get_conn():
    path = get_db_path()
    return sqlite3.connect(path, check_same_thread=False)


def query(sql, params=None):
    try:
        conn = get_conn()
        return pd.read_sql_query(sql, conn, params=params or [])
    except Exception:
        return pd.DataFrame()


def query_one(sql, params=None):
    try:
        conn = get_conn()
        cur = conn.execute(sql, params or [])
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], row))
    except Exception:
        return None


# ============================================================
# LIVE API FETCH
# ============================================================

@st.cache_data(ttl=120)
def fetch_live_markets(slug):
    """Fetch current market prices from Polymarket API."""
    try:
        resp = requests.get(f"{GAMMA_BASE}/events", params={"slug": slug}, timeout=15)
        resp.raise_for_status()
        events = resp.json()
        if not events:
            return []
        event = events[0]
        markets = []
        for m in event.get('markets', []):
            try:
                prices = json.loads(m.get('outcomePrices', '[]'))
                markets.append({
                    'question': m['question'],
                    'yes_price': float(prices[0]) if prices else None,
                    'volume': float(m.get('volume', 0) or 0),
                    'liquidity': float(m.get('liquidityNum', 0) or 0),
                    'closed': m.get('closed', False),
                    'end_date': m.get('endDate', ''),
                })
            except (json.JSONDecodeError, IndexError, TypeError):
                continue
        return [m for m in markets if not m['closed'] and m['yes_price']
                and 0.01 < m['yes_price'] < 0.99]
    except Exception:
        return []


@st.cache_data(ttl=120)
def fetch_all_live_events():
    """Fetch live data for all watched events."""
    results = {}
    for slug in WATCHED_SLUGS:
        markets = fetch_live_markets(slug)
        if markets:
            results[slug] = sorted(markets, key=lambda m: m['yes_price'])
        time.sleep(0.2)
    return results


# ============================================================
# DB DATA LOADERS
# ============================================================

def load_bankroll():
    r = query_one("SELECT * FROM bankroll WHERE id=1")
    return r or {'starting_capital': 10000, 'current_capital': 10000,
                 'total_deployed': 0, 'total_realized': 0, 'total_fees': 0,
                 'n_trades_closed': 0}


def load_open_positions():
    return query("""SELECT slug, label_short, label_long, position, kelly_fraction,
        position_size_usd, n_shares, entry_z, last_z,
        entry_price_short, entry_price_long, last_price_short, last_price_long,
        entry_spread, mtm_pnl, mtm_pnl_usd, ou_halflife, entry_ts, last_update
        FROM pair_state WHERE position != 0 ORDER BY abs(mtm_pnl_usd) DESC""")


def load_all_pairs():
    return query("""SELECT slug, label_short, label_long, last_z, last_hr, last_spread,
        ou_kappa, ou_halflife, is_cointegrated, coint_pval,
        position, kelly_fraction, position_size_usd, mtm_pnl_usd,
        n_updates, last_update FROM pair_state ORDER BY abs(last_z) DESC""")


def load_trades():
    return query("SELECT * FROM trades ORDER BY exit_ts DESC")


def load_signals(hours=72):
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    return query("SELECT * FROM signals WHERE ts >= ? ORDER BY ts DESC", [cutoff])


def load_price_history(market_question, hours_back=168):
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
    df = query("SELECT ts, yes_price FROM snapshots WHERE market_question = ? AND ts >= ? ORDER BY ts",
               [market_question, cutoff])
    if not df.empty:
        df['ts'] = pd.to_datetime(df['ts'], format='ISO8601')
    return df


def load_equity_curve():
    trades = query("SELECT exit_ts, pnl_usd FROM trades ORDER BY exit_ts")
    if trades.empty:
        return pd.DataFrame()
    br = load_bankroll()
    trades['exit_ts'] = pd.to_datetime(trades['exit_ts'], format='ISO8601')
    trades['cumulative_pnl'] = trades['pnl_usd'].cumsum()
    trades['nav'] = br['starting_capital'] + trades['cumulative_pnl']
    return trades


def load_snap_stats():
    return query_one("SELECT COUNT(*) as n, MIN(ts) as first_ts, MAX(ts) as last_ts FROM snapshots") or \
        {'n': 0, 'first_ts': None, 'last_ts': None}


# ============================================================
# CHARTS
# ============================================================

CHART_LAYOUT = dict(
    template='plotly_dark', paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
    font=dict(family="monospace"),
    margin=dict(l=0, r=0, t=30, b=0),
)


def chart_equity(trades_df, bankroll):
    if trades_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trades_df['exit_ts'], y=trades_df['nav'],
        mode='lines+markers', name='NAV',
        line=dict(color='#00ff88', width=2),
        marker=dict(size=7, color=trades_df['pnl_usd'].apply(
            lambda x: '#00ff88' if x > 0 else '#ff4444')),
    ))
    fig.add_hline(y=bankroll['starting_capital'], line_dash="dash",
                  line_color="#444", annotation_text="Starting Capital")
    fig.update_layout(**CHART_LAYOUT, height=300, title="Equity Curve",
                      yaxis_title="NAV ($)")
    return fig


def chart_trade_bars(trades_df):
    if trades_df.empty:
        return None
    df = trades_df.copy().reset_index(drop=True)
    fig = go.Figure(go.Bar(
        x=df.index, y=df['pnl_usd'],
        marker_color=df['pnl_usd'].apply(lambda x: '#00ff88' if x > 0 else '#ff4444'),
        text=df['pnl_usd'].apply(lambda x: f"${x:+.0f}"),
        textposition='outside',
        hovertext=df['slug'].str[:25],
    ))
    fig.update_layout(**CHART_LAYOUT, height=250, title="PnL per Trade",
                      yaxis_title="$", xaxis_title="Trade #")
    return fig


def chart_z_scores(pairs_df):
    if pairs_df.empty:
        return None
    df = pairs_df[pairs_df['last_z'].abs() > 0.3].head(30).copy()
    df['label'] = df['slug'].str[:25]
    df['color'] = df['last_z'].apply(
        lambda z: '#00ff88' if z < -2 else '#ff4444' if z > 2 else '#555')
    fig = go.Figure(go.Bar(
        y=df['label'], x=df['last_z'], orientation='h',
        marker_color=df['color'],
        text=df['last_z'].apply(lambda z: f"{z:.2f}"), textposition='outside',
    ))
    fig.add_vline(x=2.0, line_dash="dash", line_color="#ff444488")
    fig.add_vline(x=-2.0, line_dash="dash", line_color="#00ff8888")
    fig.update_layout(**CHART_LAYOUT, height=max(250, len(df) * 22),
                      title="Kalman Z-Scores", xaxis_title="Z-Score")
    return fig


def chart_pair_prices(label_short, label_long, hours=168):
    df_s = load_price_history(label_short, hours)
    df_l = load_price_history(label_long, hours)
    if df_s.empty or df_l.empty:
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.05,
                        subplot_titles=["Prices", "Spread"])
    fig.add_trace(go.Scatter(x=df_s['ts'], y=df_s['yes_price'],
        name=label_short[:30], line=dict(color='#ff6666', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_l['ts'], y=df_l['yes_price'],
        name=label_long[:30], line=dict(color='#00ff88', width=1.5)), row=1, col=1)
    merged = pd.merge(df_s[['ts','yes_price']], df_l[['ts','yes_price']],
                       on='ts', suffixes=('_s','_l'))
    if not merged.empty:
        merged['spread'] = merged['yes_price_l'] - merged['yes_price_s']
        fig.add_trace(go.Scatter(x=merged['ts'], y=merged['spread'],
            name='Spread', line=dict(color='#ffaa00', width=1.5),
            fill='tozeroy', fillcolor='rgba(255,170,0,0.1)'), row=2, col=1)
    fig.update_layout(**CHART_LAYOUT, height=380,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def chart_term_structure(markets):
    """Bar chart of current term structure prices."""
    if not markets:
        return None
    df = pd.DataFrame(markets).sort_values('yes_price')
    df['label'] = df['question'].str.extract(r'by (.+?)\?', expand=False).fillna(df['question'].str[:30])
    fig = go.Figure(go.Bar(
        x=df['label'], y=df['yes_price'],
        marker_color='#00aaff',
        text=df['yes_price'].apply(lambda x: f"{x:.1%}"),
        textposition='outside',
    ))
    fig.update_layout(**CHART_LAYOUT, height=250, title="Current Term Structure",
                      yaxis_title="Yes Price", yaxis_range=[0, 1])
    return fig


# ============================================================
# HELPERS
# ============================================================

def dir_badge(pos):
    if pos == 1 or pos == 'LONG_SPREAD':
        return "🟩 LONG"
    elif pos == -1 or pos == 'SHORT_SPREAD':
        return "🟥 SHORT"
    return ""


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.markdown("# 📊 Polymarket Term Structure Pairs")
    st.caption("Kalman Filter + OU Mean Reversion | Kelly Position Sizing | Live Monitoring")

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### Settings")
        auto_refresh = st.checkbox("Auto-refresh (60s)", value=True)
        show_live = st.checkbox("Fetch live prices", value=True)
        st.markdown("---")
        st.markdown("**Monitor**: run `polymarket_monitor.py --loop` locally to keep DB updating")
        st.markdown("**DB**: push updated `.db` to GitHub to refresh cloud dashboard")

    if auto_refresh:
        st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)

    # Load data
    bankroll = load_bankroll()
    positions = load_open_positions()
    all_pairs = load_all_pairs()
    trades = load_trades()
    equity = load_equity_curve()
    snap = load_snap_stats()

    nav = bankroll['current_capital'] + bankroll['total_deployed']
    unreal = positions['mtm_pnl_usd'].sum() if not positions.empty else 0
    nav += unreal
    total_return = (nav / bankroll['starting_capital']) - 1

    # ── Top Metrics ──
    c = st.columns(8)
    c[0].metric("NAV", f"${nav:,.2f}", f"{total_return:+.2%}")
    c[1].metric("Realized", f"${bankroll['total_realized']:+,.2f}")
    c[2].metric("Unrealized", f"${unreal:+,.2f}")
    c[3].metric("Cash", f"${bankroll['current_capital']:,.2f}")
    c[4].metric("Deployed", f"${bankroll['total_deployed']:,.2f}")
    c[5].metric("Positions", f"{len(positions)} open")
    c[6].metric("Trades", str(bankroll['n_trades_closed']))
    wr = (trades['pnl_usd'] > 0).mean() if not trades.empty else 0
    c[7].metric("Win Rate", f"{wr:.0%}")

    last_ts = snap['last_ts'][:19] if snap.get('last_ts') else "—"
    st.caption(f"DB last update: {last_ts} UTC | {snap.get('n', 0):,} snapshots")

    # ── Equity + PnL ──
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = chart_equity(equity, bankroll)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Equity curve appears after first closed trade.")
    with col2:
        fig = chart_trade_bars(trades)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ── Tabs ──
    tab_pos, tab_live, tab_trades, tab_signals, tab_pairs, tab_dive = st.tabs([
        "Open Positions", "Live Markets", "Trade History", "Signals", "All Pairs", "Deep Dive"
    ])

    # ── Open Positions ──
    with tab_pos:
        if positions.empty:
            st.info("No open positions.")
        else:
            for _, row in positions.iterrows():
                with st.container(border=True):
                    pc = st.columns([3, 1, 1, 1, 1, 1, 1])
                    pc[0].markdown(f"**{dir_badge(row['position'])}** `{row['slug']}`")
                    pc[1].metric("Size", f"${row['position_size_usd']:,.0f}")
                    pc[2].metric("Kelly", f"{row['kelly_fraction']:.1%}")
                    pc[3].metric("Shares", f"{row['n_shares']:,.0f}")
                    pc[4].metric("Z", f"{row['entry_z']:+.2f} → {row['last_z']:+.2f}")
                    pc[5].metric("PnL", f"${row['mtm_pnl_usd']:+,.2f}",
                                 f"{row['mtm_pnl']:+.4f}/sh")
                    held = ""
                    if row.get('entry_ts'):
                        try:
                            h = (datetime.now(timezone.utc) - datetime.fromisoformat(
                                row['entry_ts']).replace(tzinfo=timezone.utc)).total_seconds() / 3600
                            held = f"{h:.1f}h"
                        except Exception:
                            pass
                    pc[6].metric("Held", held)

                    fig = chart_pair_prices(row['label_short'], row['label_long'], 72)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

    # ── Live Markets ──
    with tab_live:
        if not show_live:
            st.info("Enable 'Fetch live prices' in sidebar.")
        else:
            with st.spinner("Fetching live Polymarket data..."):
                live = fetch_all_live_events()

            if not live:
                st.warning("Could not fetch live data from Polymarket API.")
            else:
                for slug, markets in live.items():
                    with st.expander(f"**{slug}** — {len(markets)} active markets", expanded=False):
                        # Term structure chart
                        fig = chart_term_structure(markets)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                        # Table
                        mdf = pd.DataFrame(markets)
                        mdf = mdf[['question', 'yes_price', 'volume', 'liquidity']].copy()
                        mdf.columns = ['Market', 'Yes Price', 'Volume ($)', 'Liquidity ($)']
                        mdf['Yes Price'] = mdf['Yes Price'].apply(lambda x: f"{x:.1%}")
                        mdf['Volume ($)'] = mdf['Volume ($)'].apply(lambda x: f"${x:,.0f}")
                        mdf['Liquidity ($)'] = mdf['Liquidity ($)'].apply(lambda x: f"${x:,.0f}")
                        st.dataframe(mdf, use_container_width=True, hide_index=True)

    # ── Trade History ──
    with tab_trades:
        if trades.empty:
            st.info("No closed trades yet.")
        else:
            dt = trades[['exit_ts','slug','direction','kelly_fraction',
                         'position_size_usd','n_shares','entry_z','exit_z',
                         'pnl_usd','net_pnl_per_share','hours_held','exit_reason']].copy()
            dt.columns = ['Exit','Event','Dir','Kelly','Size','Shares',
                          'Z In','Z Out','PnL ($)','PnL/sh','Hours','Reason']
            dt['Exit'] = dt['Exit'].astype(str).str[:16]
            dt['Dir'] = dt['Dir'].apply(lambda x: "LONG" if x == 'LONG_SPREAD' else "SHORT")
            dt['Kelly'] = dt['Kelly'].apply(lambda x: f"{x:.1%}")
            dt['Size'] = dt['Size'].apply(lambda x: f"${x:,.0f}")
            dt['Shares'] = dt['Shares'].apply(lambda x: f"{x:,.0f}")
            dt['PnL ($)'] = dt['PnL ($)'].apply(lambda x: f"${x:+,.2f}")
            dt['PnL/sh'] = dt['PnL/sh'].apply(lambda x: f"{x:+.4f}")
            st.dataframe(dt, use_container_width=True, hide_index=True)

            sc = st.columns(6)
            sc[0].metric("Total PnL", f"${trades['pnl_usd'].sum():+,.2f}")
            sc[1].metric("Avg PnL", f"${trades['pnl_usd'].mean():+,.2f}")
            sc[2].metric("Best", f"${trades['pnl_usd'].max():+,.2f}")
            sc[3].metric("Worst", f"${trades['pnl_usd'].min():+,.2f}")
            sc[4].metric("Avg Hold", f"{trades['hours_held'].mean():.1f}h")
            sc[5].metric("Avg Kelly", f"{trades['kelly_fraction'].mean():.1%}")

    # ── Signals ──
    with tab_signals:
        signals = load_signals(168)
        if signals.empty:
            st.info("No recent signals.")
        else:
            ds = signals[['ts','signal_type','slug','z_score',
                          'price_short','price_long','notes']].copy()
            ds.columns = ['Time','Type','Event','Z','P(short)','P(long)','Notes']
            ds['Time'] = ds['Time'].astype(str).str[:16]
            st.dataframe(ds, use_container_width=True, hide_index=True)

    # ── All Pairs ──
    with tab_pairs:
        fig_z = chart_z_scores(all_pairs)
        if fig_z:
            st.plotly_chart(fig_z, use_container_width=True)

        if not all_pairs.empty:
            da = all_pairs[['slug','last_z','last_hr','ou_halflife',
                            'is_cointegrated','position','kelly_fraction',
                            'position_size_usd','mtm_pnl_usd','n_updates']].copy()
            da.columns = ['Event','Z','HR','OU t½','Coint','Pos','Kelly',
                          'Size ($)','MTM ($)','Updates']
            da['Pos'] = da['Pos'].apply(lambda x: "LONG" if x==1 else "SHORT" if x==-1 else "—")
            da['Coint'] = da['Coint'].apply(lambda x: "Yes" if x==1 else "No")
            da['Kelly'] = da['Kelly'].apply(lambda x: f"{x:.1%}" if x else "—")
            st.dataframe(da, use_container_width=True, hide_index=True)

    # ── Deep Dive ──
    with tab_dive:
        if all_pairs.empty:
            st.info("No pairs data.")
        else:
            slugs = all_pairs['slug'].unique().tolist()
            sel = st.selectbox("Select event", slugs)
            slug_pairs = all_pairs[all_pairs['slug'] == sel]

            pair_labels = []
            for _, row in slug_pairs.iterrows():
                for lbl in [row.get('label_short'), row.get('label_long')]:
                    if lbl and lbl not in pair_labels:
                        pair_labels.append(lbl)

            if len(pair_labels) >= 2:
                lookback = st.slider("Lookback (hours)", 24, 336, 168, 24)
                colors = ['#00ff88', '#ff6666', '#ffaa00', '#44aaff', '#ff44ff']
                fig = go.Figure()
                for i, label in enumerate(pair_labels[:5]):
                    df = load_price_history(label, lookback)
                    if not df.empty:
                        fig.add_trace(go.Scatter(
                            x=df['ts'], y=df['yes_price'],
                            name=label[:40],
                            line=dict(color=colors[i % len(colors)], width=1.5)))
                fig.update_layout(**CHART_LAYOUT, height=400, yaxis_title="Yes Price",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig, use_container_width=True)

            for _, row in slug_pairs.iterrows():
                z_icon = "🟢" if abs(row['last_z']) > 2 else "⚪"
                pos = dir_badge(row['position']) if row['position'] != 0 else ""
                hl = f"{row['ou_halflife']:.1f}" if row.get('ou_halflife') else "—"
                st.markdown(
                    f"{z_icon} **z={row['last_z']:+.2f}** | "
                    f"HR={row['last_hr']:.3f} | OU t½={hl} | "
                    f"{'✓ Cointegrated' if row.get('is_cointegrated') else ''} {pos}")

    # ── Footer ──
    st.markdown("---")
    st.caption("Polymarket Term Structure Pairs Trading | "
               "Kalman Filter + OU Mean Reversion + Kelly Sizing | "
               "[GitHub](https://github.com/razr321/polymarket-pairs)")


if __name__ == "__main__":
    main()
