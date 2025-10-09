import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import sys, io
import argparse

parser = argparse.ArgumentParser(description="Recieve ticker symbol")
parser.add_argument("--ticker", required=True)
args = parser.parse_args()

ticker = args.ticker

# Ensure stdout can print unicode before importing the SMC package (which prints a banner)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass
# Import SMC indicators (installed package API exposes classmethods on smc.smc)
from smartmoneyconcepts.smc import smc

# --------------------------
# 1. Load Weekly OHLC data
# --------------------------
df = pd.read_csv(f"ohlc_weekly_data/{ticker}_Weekly.csv")
df.reset_index(inplace=True)
# Normalize any MultiIndex/tuple columns into simple string names before further processing
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ["_".join([str(x) for x in col if x is not None]) for col in df.columns]
else:
    df.columns = ["_".join([str(x) for x in col]) if isinstance(col, tuple) else str(col) for col in df.columns]
df.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}, inplace=True)
# Ensure required OHLCV columns exist in lowercase
for src, dst in [("Open","open"),("High","high"),("Low","low"),("Close","close"),("Volume","volume")]:
    if dst not in df.columns and src in df.columns:
        df[dst] = df[src]
for dst in ["open","high","low","close","volume"]:
    if dst not in df.columns:
        # Backfill missing numeric columns with NaN/0 where sensible
        df[dst] = 0 if dst == "volume" else pd.NA

# --------------------------
# 2. Compute SMC Indicators
# --------------------------
# Swing Highs & Lows (foundation for several other signals)
shl = smc.swing_highs_lows(df, swing_length=3)

# BOS + CHoCH (levels and broken indices)
bos_choch = smc.bos_choch(df, shl, close_break=True)

# Order Blocks
obs = smc.ob(df, shl, close_mitigation=False)

# Fair Value Gaps
fvgs = smc.fvg(df, join_consecutive=False)

# Liquidity
liq = smc.liquidity(df, shl, range_percent=0.01)

# --------------------------
# 2.5 Merge indicators into OHLC and export CSV
# --------------------------
def _merge_with_prefix(base_df: pd.DataFrame, indicator_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Concatenate indicator columns to base_df with a prefix, aligned by index."""
    if indicator_df is None or len(indicator_df) == 0:
        return base_df
    aligned_base = base_df.reset_index(drop=True)
    aligned_ind = indicator_df.reset_index(drop=True).add_prefix(f"{prefix}_")
    return pd.concat([aligned_base, aligned_ind], axis=1)

for _prefix, _ind in [("shl", shl), ("bos", bos_choch), ("ob", obs), ("fvg", fvgs), ("liq", liq)]:
    df = _merge_with_prefix(df, _ind, _prefix)

# Save merged OHLC + indicators to CSV
try:
    df.to_csv(f"data/{ticker}_SMC_merged.csv", index=False)
except Exception:
    # Fallback to project root if data/ doesn't exist
    df.to_csv(f"{ticker}_SMC_merged.csv", index=False)

# --------------------------
# 3. Plot with Plotly
# --------------------------
# fig = go.Figure()

# # Candlestick base
# fig.add_trace(go.Candlestick(
#     x=df['date'],
#     open=df['open'], high=df['high'],
#     low=df['low'], close=df['close'],
#     name="Price"
# ))

# # Add BOS / CHoCH markers at their levels
# for i, row in bos_choch.iterrows():
#     if not pd.isna(row.get('BOS')):
#         fig.add_trace(go.Scatter(
#             x=[df['date'].iloc[i]], y=[row['Level']],
#             mode="markers+text",
#             text=["BOS"],
#             textposition="top center",
#             marker=dict(color="blue", size=10, symbol="triangle-up"),
#             name="BOS"
#         ))
#     if not pd.isna(row.get('CHOCH')):
#         fig.add_trace(go.Scatter(
#             x=[df['date'].iloc[i]], y=[row['Level']],
#             mode="markers+text",
#             text=["CHoCH"],
#             textposition="bottom center",
#             marker=dict(color="orange", size=10, symbol="triangle-down"),
#             name="CHoCH"
#         ))

# # Plot Swing Highs / Swing Lows from swing_highs_lows output
# for i, row in shl.iterrows():
#     if row['HighLow'] == 1:
#         fig.add_trace(go.Scatter(
#             x=[df['date'].iloc[i]], y=[row['Level']],
#             mode="markers", marker=dict(color="red", size=8, symbol="triangle-down"), name="Swing High"
#         ))
#     elif row['HighLow'] == -1:
#         fig.add_trace(go.Scatter(
#             x=[df['date'].iloc[i]], y=[row['Level']],
#             mode="markers", marker=dict(color="green", size=8, symbol="triangle-up"), name="Swing Low"
#         ))

# # Plot Order Blocks (as shaded rectangles from OB candle to mitigation)
# for i, row in obs.iterrows():
#     if not pd.isna(row['OB']):
#         x0 = df['date'].iloc[i]
#         x1 = df['date'].iloc[int(row['MitigatedIndex'])] if not pd.isna(row['MitigatedIndex']) and int(row['MitigatedIndex']) < len(df) else x0
#         fig.add_shape(
#             type="rect",
#             x0=x0, x1=x1,
#             y0=row['Bottom'], y1=row['Top'],
#             fillcolor="purple", opacity=0.2, line_width=0
#         )

# # Plot FVGs
# for i, row in fvgs.iterrows():
#     if not pd.isna(row['FVG']):
#         x0 = df['date'].iloc[i]
#         x1 = df['date'].iloc[int(row['MitigatedIndex'])] if not pd.isna(row['MitigatedIndex']) and int(row['MitigatedIndex']) < len(df) else x0
#         fig.add_shape(
#             type="rect",
#             x0=x0, x1=x1,
#             y0=row['Bottom'], y1=row['Top'],
#             fillcolor="yellow", opacity=0.2, line_width=0
#         )

# # Add Liquidity levels
# for i, row in liq.iterrows():
#     if not pd.isna(row['Liquidity']):
#         fig.add_trace(go.Scatter(
#             x=[df['date'].iloc[i]], y=[row['Level']],
#             mode="markers", marker=dict(color="cyan", size=8, symbol="x"),
#             name="Liquidity"
#         ))

# # Layout tweaks
# fig.update_layout(
#     title=f"{ticker} - Smart Money Concepts (Weekly)",
#     xaxis_title="Date",
#     yaxis_title="Price",
#     template="plotly_dark",
#     xaxis_rangeslider_visible=False,
#     height=900
# )

# fig.write_html(f"plots/{ticker}_SMC_plot.html")