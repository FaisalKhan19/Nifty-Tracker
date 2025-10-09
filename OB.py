import pandas as pd
import plotly.graph_objects as go
import os

def detect_all_ob(df:pd.DataFrame, wick_thresh:float = 1.025, vol_thresh:float = 1.30, pivot_len:int = 5):
    # Data sanity check
    expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in expected_columns):
        print(f"Expected columns: {expected_columns}")
        print(f"Found columns: {list(df.columns)}")
        raise ValueError("CSV must contain columns: date, open, high, low, close, volume")
    
    df = df.reset_index(drop=True)

    for i, row in df.loc[pivot_len:].iterrows():
        vol_past_avg = df.loc[i-pivot_len:i-1, 'volume'].mean()
        vol_next_avg = df.loc[i+1: i+pivot_len, 'volume'].mean()
        vol_curr = row['volume']
        if (vol_curr/vol_past_avg > vol_thresh) and (vol_curr/vol_next_avg > vol_thresh):
            wick_lower = row['open']/row['low']
            wick_upper = row['high']/row['close']
            # Bearish OB
            if wick_upper > wick_lower and wick_upper > wick_thresh:
                zone_high = row['high']
                zone_low = row['close']
                future_idx = df.index[i:]  # indices from i onward
                mask = df.loc[future_idx, 'close'] > zone_high
                mitigated = future_idx[mask]

                if len(mitigated) > 0:
                    mitigated_index = mitigated[0]
                else:
                    mitigated_index = df.index[-1]

                df.loc[i, 'ob_type'] = 'bearish'
                df.loc[i, 'ob_zone_low'] = zone_low
                df.loc[i, 'ob_zone_high'] = zone_high
                df.loc[i, 'ob_mitigated_idx'] = mitigated_index

            # Bullish OB
            if wick_lower > wick_upper and wick_lower > wick_thresh:
                zone_high = row['close']
                zone_low = row['low']
                future_idx = df.index[i:]  # indices from i onward
                mask = df.loc[future_idx, 'close'] < zone_low
                mitigated = future_idx[mask]

                if len(mitigated) > 0:
                    mitigated_index = mitigated[0]   # first mitigation candle
                else:
                    mitigated_index = df.index[-1]   # fallback: last candle

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

def make_plots(base_path:str = 'ohlc_weekly_data', save_path:str = 'OB_plots'):
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