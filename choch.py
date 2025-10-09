import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse

parser = argparse.ArgumentParser(description="Recieve ticker symbol")
parser.add_argument("--ticker", required=True)
args = parser.parse_args()

ticker = args.ticker
df = pd.read_csv(os.path.join("data", f"{ticker}_SMC_merged.csv"))
# Ensure date is datetime and sorted for plotting
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

# df = df.drop(['index', 'symbol', 'sma', 'diff', 'diff/avg_diff', 'momentum',
#        'delta_m', 'CF', 'DDF', 'TSR', 'Class',  'ob_OB',
#        'ob_Top', 'ob_Bottom', 'ob_OBVolume', 'ob_MitigatedIndex',
#        'ob_Percentage', 'fvg_FVG', 'fvg_Top', 'fvg_Bottom',
#        'fvg_MitigatedIndex', 'liq_Liquidity', 'liq_Level', 'liq_End',
#        'liq_Swept'], axis=1)

bullish_pattern = [-1, 1, -1, 1]  # Bullish CoC+
bearish_pattern = [1, -1, 1, -1]  # Bearish CoC+

l = 20  # Lookback
df["CoC+"] = np.zeros(len(df))
for i in range(l, len(df)):
    # Use series to retain original indices for swings
    shl_series = df["shl_HighLow"].iloc[i - l : i].dropna()
    level_series = df["shl_Level"].iloc[i - l : i].dropna()
    # Use closes at the swing indices for CoC comparisons
    close_series = df.loc[shl_series.index, "close"]
    hist = shl_series.to_list()
    levels = level_series.to_list()
    closes = close_series.to_list()
    # Need at least 5 swings to evaluate the pattern logic safely
    if len(hist) < 5:
        continue
    if hist[-4:] == bullish_pattern:
        # Check Break of Structure with bearish expectancy and bullish CoC
        if (
            (levels[-5] > levels[-3])
            and (levels[-4] < levels[-2])
            and (levels[-3] < levels[-1])
        ):
            # Place marker at the second most recent swing's actual row index
            swing_index = shl_series.index[-2]
            df.loc[swing_index, "CoC+"] = 1
            df.loc[swing_index, "level"] = level_series.iloc[-2]
    if hist[-4:] == bearish_pattern:
        # Check Break of Structure with bullish expectancy and bearish CoC
        if (
            (levels[-5] < levels[-3])
            and (levels[-4] > levels[-2])
            and (levels[-3] > levels[-1])
        ):
            # Place marker at the second most recent swing's actual row index
            swing_index = shl_series.index[-2]
            df.loc[swing_index, "CoC+"] = -1
            df.loc[swing_index, "level"] = level_series.iloc[-2]


# Assuming you already have your dataframe 'df' with the CoC+ signals
# Let's create a visualization
df.index = pd.to_datetime(df["date"])
df.to_csv(f"plots_data/{ticker}.csv", index=False)
# Create subplots with 2 rows: one for price and CoC+ signals, one for volume (optional)
fig = make_subplots(
    rows=1,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=("Price with CHoCH+ Signals"),
    row_width=[0.2],
)

# Add candlestick chart
fig.add_trace(
    go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price",
    ),
    row=1,
    col=1,
)

# Add bullish CHoCH+ signals (value = 1)
bullish_signals = df[df["CoC+"] == 1]
fig.add_trace(
    go.Scatter(
        x=bullish_signals["date"],
        y=bullish_signals["level"],
        mode="markers",
        marker=dict(color="green", size=15, symbol="triangle-up"),
        name="Bullish CHoCH+",
        hovertemplate="Bullish CHoCH+<br>Time: %{x}<br>Level: %{y}<extra></extra>",
    ),
    row=1,
    col=1,
)

# Add bearish CHoCH+ signals (value = -1)
bearish_signals = df[df["CoC+"] == -1]
fig.add_trace(
    go.Scatter(
        x=bearish_signals["date"],
        y=bearish_signals["level"],
        mode="markers",
        marker=dict(color="red", size=15, symbol="triangle-down"),
        name="Bearish CHoCH+",
        hovertemplate="Bearish CHoCH+<br>Time: %{x}<br>Level: %{y}<extra></extra>",
    ),
    row=1,
    col=1,
)

# Add swing markers (all swings in history)
swing_highs = df[df["shl_HighLow"] == 1]
fig.add_trace(
    go.Scatter(
        x=swing_highs["date"],
        y=swing_highs["shl_Level"],
        mode="markers",
        marker=dict(color="blue", size=7, symbol="triangle-up"),
        name="Swing High",
        hovertemplate="Swing High<br>Time: %{x}<br>Level: %{y}<extra></extra>",
    ),
    row=1,
    col=1,
)

swing_lows = df[df["shl_HighLow"] == -1]
fig.add_trace(
    go.Scatter(
        x=swing_lows["date"],
        y=swing_lows["shl_Level"],
        mode="markers",
        marker=dict(color="orange", size=7, symbol="triangle-down"),
        name="Swing Low",
        hovertemplate="Swing Low<br>Time: %{x}<br>Level: %{y}<extra></extra>",
    ),
    row=1,
    col=1,
)

# Plot BOS / CHoCH markers similar to smc_plot.py
for i in range(len(df)):
    # BOS
    bos_val = df['bos_BOS'].iloc[i] if 'bos_BOS' in df.columns else np.nan
    if not pd.isna(bos_val):
        level = df['bos_Level'].iloc[i] if 'bos_Level' in df.columns else np.nan
        if not pd.isna(level):
            fig.add_trace(go.Scatter(
                x=[df['date'].iloc[i]], y=[level],
                mode="markers+text",
                text=["BOS"], textposition="top center",
                marker=dict(color="blue", size=10, symbol="triangle-up"),
                name="BOS"
            ), row=1, col=1)
    # CHoCH
    choch_val = df['bos_CHOCH'].iloc[i] if 'bos_CHOCH' in df.columns else np.nan
    if not pd.isna(choch_val):
        level = df['bos_Level'].iloc[i] if 'bos_Level' in df.columns else np.nan
        if not pd.isna(level):
            fig.add_trace(go.Scatter(
                x=[df['date'].iloc[i]], y=[level],
                mode="markers+text",
                text=["CHoCH"], textposition="bottom center",
                marker=dict(color="orange", size=10, symbol="triangle-down"),
                name="CHoCH"
            ), row=1, col=1)

# Plot Order Blocks as rectangles
if all(col in df.columns for col in ['ob_OB', 'ob_Top', 'ob_Bottom']):
    for i in range(len(df)):
        if not pd.isna(df['ob_OB'].iloc[i]):
            y0 = df['ob_Bottom'].iloc[i]
            y1 = df['ob_Top'].iloc[i]
            if pd.isna(y0) or pd.isna(y1):
                continue
            x0 = df['date'].iloc[i]
            x1 = x0
            if 'ob_MitigatedIndex' in df.columns and not pd.isna(df['ob_MitigatedIndex'].iloc[i]):
                mi = int(df['ob_MitigatedIndex'].iloc[i])
                if 0 <= mi < len(df):
                    x1 = df['date'].iloc[mi]
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                          fillcolor="purple", opacity=0.2, line_width=0)

# Plot FVGs as rectangles
# Plot FVGs
for i, row in df.iterrows():
    if not pd.isna(row['fvg_FVG']):
        x0 = row['date']
        x1 = df['date'].iloc[int(row['fvg_MitigatedIndex'])] if not pd.isna(row['fvg_MitigatedIndex']) and int(row['fvg_MitigatedIndex']) < len(df) else x0
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=row['fvg_Bottom'], y1=row['fvg_Top'],
            fillcolor="yellow", opacity=0.2, line_width=0
        )

# # Plot Liquidity markers
# if 'liq_Liquidity' in df.columns:
#     for i in range(len(df)):
#         if not pd.isna(df['liq_Liquidity'].iloc[i]):
#             level = df['liq_Level'].iloc[i] if 'liq_Level' in df.columns else np.nan
#             if not pd.isna(level):
#                 fig.add_trace(go.Scatter(
#                     x=[df['date'].iloc[i]], y=[level],
#                     mode="markers",
#                     marker=dict(color="cyan", size=8, symbol="x"),
#                     name="Liquidity"
#                 ), row=1, col=1)
# Update layout
fig.update_layout(
    title="Price Chart with CHoCH+ Signals",
    yaxis_title="Price",
    xaxis_title="Date",
    template="plotly_white",
    height=800,
    showlegend=True,
)

# Update x-axis rangeslider
fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

fig.write_html(f"plots/{ticker}_smc_plot.html")
