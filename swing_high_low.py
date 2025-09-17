import pandas as pd
import os
import plotly.graph_objects as go
import ta
ohlc_path = "ohlc_weekly_data"

df = pd.read_csv(os.path.join(ohlc_path, "INDUSINDBK_weekly.csv"))
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
df = df.rename(columns={'open': "Open", 'high': 'High', 'low': 'Low', 'close': 'Close'})

# Simple moving average (20-period by default)
df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()

df['isSL'] = 0
df['isSH'] = 0
df['detection_time'] = 0
pre_S_type = 'n'
pre_SH = 0
pre_SL = 0
pre_SL_index = -1
pre_SH_index = -1


## Function to detect swing lows and swing highs with regards to the instructions described by Lance Beggs.
def which_S(df, j):
    global pre_S_type
    global pre_SL
    global pre_SH
    global pre_SL_index
    global pre_SH_index
    
    support = (
                ((df['Low'][j-3] < df['Low'][j-2]) and (df['Low'][j-3] < df['Low'][j-1]) and \
                (df['Low'][j-3] < df['Low'][j-4]) and (df['Low'][j-3] < df['Low'][j-5])) or \
               (
                ((abs(df['Low'][j-3]-df['Low'][j-4]) < 10) and (df['Low'][j-2]>df['Low'][j-3])\
                 and (df['Low'][j-1]>df['Low'][j-3]) and \
                (df['Low'][j-5]>df['Low'][j-3]) and (df['Low'][j-6]>df['Low'][j-3])) or \
                    
                ((abs(df['Low'][j-3]-df['Low'][j-4]) < 10) and (abs(df['Low'][j-3]-df['Low'][j-5]) < 10)\
                 and (df['Low'][j-2]>df['Low'][j-3]) and (df['Low'][j-1]>df['Low'][j-3]) and \
                (df['Low'][j-6]>df['Low'][j-3]) and (df['Low'][j-7]>df['Low'][j-3])) or \
                
                ((abs(df['Low'][j-3]-df['Low'][j-5]) < 10) and (df['Low'][j-4]>df['Low'][j-3]) \
                 and (df['Low'][j-2]>df['Low'][j-3]) and (df['Low'][j-1]>df['Low'][j-3]) and \
                (df['Low'][j-6]>df['Low'][j-3]) and (df['Low'][j-7]>df['Low'][j-3]))
              )
    )
    
    resistance = (
                ((df['High'][j-3] > df['High'][j-2]) and (df['High'][j-3] > df['High'][j-1]) and \
                (df['High'][j-3] > df['High'][j-4]) and (df['High'][j-3] > df['High'][j-5])) or \
               (
                ((abs(df['High'][j-3]-df['High'][j-4]) < 10) and (df['High'][j-2]<df['High'][j-3]) and \
                 (df['High'][j-1]<df['High'][j-3]) and \
                (df['High'][j-5]<df['High'][j-3]) and (df['High'][j-6]<df['High'][j-3])) or \
                    
                ((abs(df['High'][j-3]-df['High'][j-4]) < 10) and (abs(df['High'][j-3]-df['High'][j-5]) < 10) and \
                 (df['High'][j-2]<df['High'][j-3]) and (df['High'][j-1]<df['High'][j-3]) and \
                (df['High'][j-6]<df['High'][j-3]) and (df['High'][j-7]<df['High'][j-3])) or \
                
                ((abs(df['High'][j-3]-df['High'][j-5]) < 10) and (df['High'][j-4]<df['High'][j-3]) and \
                 (df['High'][j-2]<df['High'][j-3]) and (df['High'][j-1]<df['High'][j-3]) and \
                (df['High'][j-6]<df['High'][j-3]) and (df['High'][j-7]<df['High'][j-3]))
              )
    )
    if support:
        if pre_S_type=='n' or pre_S_type=='H':
            pre_S_type = 'L'
            pre_SL = df['Low'][j-3]
            pre_SL_index = j-3
            df['isSL'][j-3] = 1
            to_append = [df['date'][j], df['date'][j-3], df['Low'][j-3], j-3]
            SLs_length = len(SLs)
            SLs.loc[SLs_length] = to_append
            df['detection_time'][j-3] = df['date'][j]
                       
        elif(pre_S_type=='L' and df['Low'][j-3]<pre_SL):
            pre_SL = df['Low'][j-3]
            df['isSL'][pre_SL_index] = 0
            df['detection_time'][pre_SL_index] = 0
            df['isSL'][j-3] = 1
            df['detection_time'][j-3] = df['date'][j]
            SLs['SL_value'][df.index[pre_SL_index]] = df['Low'][j-3]
            SLs['CandleNumber'][df.index[pre_SL_index]] = j-3
            pre_SL_index = j-3       
#####################################################################################################        
    elif resistance:
        if pre_S_type=='n' or pre_S_type=='L':
            pre_S_type = 'H'
            pre_SH = df['High'][j-3]
            pre_SH_index = j-3
            df['isSH'][j-3] = 1
            to_append = [df['date'][j], df['date'][j-3], df['High'][j-3], j-3]
            SHs_length = len(SHs)
            SHs.loc[SHs_length] = to_append
            df['detection_time'][j-3] = df['date'][j]
            
        elif (pre_S_type=='H' and df['High'][j-2]>pre_SH):
            pre_SH = df['High'][j-3]
            df['isSH'][pre_SH_index] = 0
            df['detection_time'][pre_SH_index] = 0
            df['isSH'][j-3] = 1
            df['detection_time'][j-3] = df['date'][j]
            SHs['SH_value'][df.index[pre_SH_index]] = df['High'][j-3]
            SHs['CandleNumber'][df.index[pre_SH_index]] = j-3
            pre_SH_index = j-3
            
## Variables to store swing lows and swing highs.
SHs = pd.DataFrame(columns=['date_know', 'date_actu', 'SH_value', 'CandleNumber'])
SLs = pd.DataFrame(columns=['date_know', 'date_actu', 'SL_value', 'CandleNumber'])

## This loop adds swing highs and swing lows with their detection time to the original dataframe.
for i in range(8, df.shape[0]-2):
    which_S(df, i)

# Parameters
lookback = 10  # bars before
lookahead = 10 # bars after

# Detect swing highs/lows
swing_highs = []
swing_lows = []

for i in range(lookback, len(df) - lookahead):
    high = df.loc[i, "High"]
    low = df.loc[i, "Low"]

    if high == df["High"].iloc[i - lookback : i + lookahead + 1].max():
        swing_highs.append((df.loc[i, "date"], high))
    elif low == df["Low"].iloc[i - lookback : i + lookahead + 1].min():
        swing_lows.append((df.loc[i, "date"], low))

# Convert to DataFrame
SHs = pd.DataFrame(swing_highs, columns=["date", "SH_value"])
SLs = pd.DataFrame(swing_lows, columns=["date", "SL_value"])

# Make sure date types align
df['date'] = pd.to_datetime(df['date'])
SHs['date'] = pd.to_datetime(SHs['date'])
SLs['date'] = pd.to_datetime(SLs['date'])

# Merge with left join
df = df.merge(SHs, on='date', how='left')
df = df.merge(SLs, on='date', how='left')

# Optional: add flags
df['isSH'] = df['SH_value'].notna().astype(int)
df['isSL'] = df['SL_value'].notna().astype(int)

# --- Plot ---
fig = go.Figure(data=[go.Candlestick(
    x=df['date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="Candles"
)])

# Add VWAP overlay
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['vwap'],
    mode='lines',
    line=dict(color='orange', width=2),
    name='VWAP'
))

# Add SMA overlay
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['SMA_20'],
    mode='lines',
    line=dict(color='deepskyblue', width=2),
    name='SMA 20'
))

# Plot swing highs
fig.add_trace(go.Scatter(
    x=SHs['date'],
    y=SHs['SH_value'],
    mode="markers+text",
    marker=dict(symbol="triangle-down", size=14, color="red"),
    text=["SH"]*len(SHs),
    textposition="top center",
    name="Swing Highs"
))

# Plot swing lows
fig.add_trace(go.Scatter(
    x=SLs['date'],
    y=SLs['SL_value'],
    mode="markers+text",
    marker=dict(symbol="triangle-up", size=14, color="lime"),
    text=["SL"]*len(SLs),
    textposition="bottom center",
    name="Swing Lows"
))

fig.update_layout(
    title="Confirmed Swing Highs & Lows (using future data)",
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=800
)
df.to_csv("swing_HL.csv", index=False)
# Save
fig.write_html("zigzag_swings.html")
print("✅ Saved as zigzag_swings.html")
