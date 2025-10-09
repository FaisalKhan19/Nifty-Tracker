import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

def detect_all_ob(df: pd.DataFrame, atr_len: int = 5, wick_mult: float = 0.25, vol_len: int = 5, vol_z: float = 1.5, pivot_len: int = 5):
    # Data sanity check
    expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"CSV must contain columns: {expected_columns}")

    df = df.reset_index(drop=True)

    # --- ATR (volatility proxy) ---
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(atr_len).mean()

    # --- Wick sizes ---
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']).clip(lower=0)
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)).clip(lower=0)

    # --- Volume z-score ---
    df['vol_mean'] = df['volume'].rolling(vol_len).mean()
    df['vol_std'] = df['volume'].rolling(vol_len).std()
    df['vol_z'] = (df['volume'] - df['vol_mean']) / df['vol_std']

    # --- Iterate for OB detection ---
    for i, row in df.loc[pivot_len:].iterrows():
        if pd.isna(row['atr']) or pd.isna(row['vol_z']):
            continue

        # Volume anomaly check
        if row['vol_z'] < vol_z:
            continue

        # --- Bearish OB ---
        if row['upper_wick'] > wick_mult * row['atr']:
            zone_high = row['high']
            zone_low = row['close']
            future_idx = df.index[i:]
            mask = df.loc[future_idx, 'close'] > zone_high
            mitigated = future_idx[mask]
            mitigated_index = mitigated[0] if len(mitigated) > 0 else df.index[-1]

            df.loc[i, 'ob_type'] = 'bearish'
            df.loc[i, 'ob_zone_low'] = zone_low
            df.loc[i, 'ob_zone_high'] = zone_high
            df.loc[i, 'ob_mitigated_idx'] = mitigated_index

        # --- Bullish OB ---
        if row['lower_wick'] > wick_mult * row['atr']:
            zone_high = row['close']
            zone_low = row['low']
            future_idx = df.index[i:]
            mask = df.loc[future_idx, 'close'] < zone_low
            mitigated = future_idx[mask]
            mitigated_index = mitigated[0] if len(mitigated) > 0 else df.index[-1]

            df.loc[i, 'ob_type'] = 'bullish'
            df.loc[i, 'ob_zone_low'] = zone_low
            df.loc[i, 'ob_zone_high'] = zone_high
            df.loc[i, 'ob_mitigated_idx'] = mitigated_index

    return df

def plot_order_blocks(df: pd.DataFrame, html_file: str = "order_blocks.html"):
    fig = go.Figure()

    # --- Candlesticks ---
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Candles"
    ))
    if 'ob_type' not in df.columns:
        return
    # --- OB Rectangles ---
    for i, row in df.dropna(subset=['ob_type']).iterrows():
        x0 = row['date']
        x1 = df.loc[int(row['ob_mitigated_idx']), 'date'] if not pd.isna(row['ob_mitigated_idx']) else df['date'].iloc[-1]
        y0 = row['ob_zone_low']
        y1 = row['ob_zone_high']

        if row['ob_type'] == 'bullish':
            color = "rgba(0,200,0,0.3)"
            line_color = "green"
        else:
            color = "rgba(200,0,0,0.3)"
            line_color = "red"

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            fillcolor=color,
            line=dict(color=line_color, width=1),
            layer="below"
        )

    # --- Layout ---
    fig.update_layout(
        title="Order Block Zones",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=900
    )

    # Save as HTML
    fig.write_html(html_file)

def make_plots(base_path:str = 'ohlc_weekly_data', save_path:str = 'OB_plots_v1'):
    os.makedirs(save_path, exist_ok=True)
    for file in os.listdir(base_path):
        if not file.endswith('csv'):
            continue
        ticker = file.split('_')[0]
        df = pd.read_csv(os.path.join(base_path, file))
        df = detect_all_ob(df)
        plot_order_blocks(df, os.path.join(save_path, f"{ticker}.html"))

if __name__ == "__main__":
    make_plots()
    # ticker = 'GAIL'
    # df = pd.read_csv(f"ohlc_weekly_data/{ticker}_weekly.csv")
    # df = detect_all_ob(df)
    # plot_order_blocks(df, f"{ticker}_OB_chart.html")