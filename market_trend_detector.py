import pandas as pd
import os
import plotly.graph_objects as go

ohlc_path = "ohlc_weekly_data"

df = pd.read_csv(os.path.join(ohlc_path, "INDUSINDBK_weekly.csv"))

# --- Step 1: SMA ---
window = 20
df['sma'] = df['close'].rolling(window).mean()

# --- Step 2: Diff from SMA ---
df['diff'] = df.apply(
    lambda row: abs(row['close']/row['sma'] - 1)*100 if row['close'] > row['sma'] 
                else abs(row['sma']/row['close'] - 1)*100,
    axis=1
)
df['avg_diff'] = df['diff'].shift(1).rolling(window-2).mean()
df['diff_over_avg'] = df['diff'] / df['avg_diff']

# --- Step 3: Momentum + ΔMomentum ---
df['momentum'] = abs(df['close'] - df['sma']) / df['sma']
df['delta_momentum'] = df['momentum'].diff()

# Continuation & Die down
N = 3
df['CF'] = df['delta_momentum'].clip(lower=0).rolling(N).sum()
df['DF'] = (-df['delta_momentum']).clip(lower=0).rolling(N).sum()
df['TSR'] = (df['CF'] / (df['DF']*0.65 + 1e-2)).clip(upper=10)
df['TSR'] = df['TSR'].rolling(4).mean()

# --- Step 4: Swing classification ---
def swing_label(row):
    if row['diff_over_avg'] > 2:
        if row['close'] > row['open']:
            return "bullish"
        elif row['close'] < row['open']:
            return "bearish"
    return None

df['swing'] = df.apply(swing_label, axis=1)

# --- Step 5: Market regime propagation ---
regime = None
trend_states = []
for i, row in df.iterrows():
    if row['swing'] is not None:
        # new swing overrides old regime
        regime = row['swing']
    elif regime is not None:
        # continue regime while TSR supports it
        if regime == "bullish" and row['TSR'] < 0.6:
            regime = None
        elif regime == "bearish" and row['TSR'] < 0.6:
            regime = None
    trend_states.append(regime if regime is not None else "neutral")

df['market_regime'] = trend_states

# --- Final output ---
print(df[['date','open','close','sma','diff_over_avg','swing','TSR','market_regime']])
df.to_csv("trend_detection.csv", index=False)

df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

# --- Candlestick chart ---
fig = go.Figure(data=[go.Candlestick(
    x=df['date'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name="Price"
)])

# --- Background shading by regime ---
current_regime = None
start_idx = None

for i, row in df.iterrows():
    if row['market_regime'] != current_regime:
        if current_regime is not None:
            fig.add_vrect(
                x0=df.loc[start_idx, 'date'],
                x1=row['date'],
                fillcolor=(
                    "rgba(0,200,0,0.30)" if current_regime == "bullish"
                    else "rgba(200,0,0,0.30)" if current_regime == "bearish"
                    else "rgba(255,215,0,0.30)"
                ),
                layer="below", line_width=0
            )
        current_regime = row['market_regime']
        start_idx = i

# Close last regime block
if current_regime is not None:
    fig.add_vrect(
        x0=df.loc[start_idx, 'date'],
        x1=df['date'].iloc[-1],
        fillcolor=(
            "rgba(0,200,0,0.35)" if current_regime == "bullish"
            else "rgba(200,0,0,0.35)" if current_regime == "bearish"
            else "rgba(255,215,0,0.35)"
        ),
        layer="below", line_width=0
    )

# --- Layout fix ---
fig.update_layout(
    title="Market Regime Candlestick Chart",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    xaxis=dict(range=[df['date'].min(), df['date'].max()]),  # ✅ full spread
    template="plotly_white"
)

# --- Save to HTML ---
fig.write_html("market_regime_chart.html")

print("✅ Chart saved as market_regime_chart.html (with better colors & full width)")
