import os
import json
import numpy as np
import pandas as pd

def detect_order_blocks(df: pd.DataFrame, lookahead: int = 3, displacement_factor: float = 0.075):
    """
    Detect bullish and bearish order blocks in OHLCV data.
    
    Parameters:
        df : DataFrame with columns ['open','high','low','close']
        lookahead : int, how many candles ahead to confirm displacement
        displacement_factor : float, min body size multiple vs avg body to confirm displacement
    
    Returns:
        List of dicts with OB info
    """
    order_blocks = []
    df['candle_type'] = np.where(df['open'] > df['close'], -1, 1)
    # Calculate average body size for displacement check
    avg_body = (df['close'] - df['open']).abs().rolling(20).mean()
    
    for i in range(len(df) - lookahead):
        d, o, h, l, c = df.loc[i, ['date', 'open','high','low','close']]
        c_3 = df.loc[min(i+1+lookahead, len(df)-1), 'close']
        body = abs(c - o)
        
        # Bullish OB = last red candle before green drive
        if c < o:  # red candle
            forward_closes = df['close'].iloc[i+1:i+1+lookahead]
            forward_cdl_types = df['candle_type'].iloc[i+1:i+1+lookahead].sum()
            if all(forward_closes > c) and ((c_3 / c - 1)> displacement_factor) and (forward_cdl_types==lookahead):
                order_blocks.append({
                    "type": "bullish",
                    "date": d,
                    "index": i,
                    "zone_low": l,
                    "zone_high": h
                })
        
        # Bearish OB = last green candle before red drive
        if c > o:  # green candle
            forward_closes = df['close'].iloc[i+1:i+1+lookahead]
            forward_cdl_types = df['candle_type'].iloc[i+1:i+1+lookahead].sum()
            if all(forward_closes < c) and ((c / c_3 - 1) > displacement_factor) and (forward_cdl_types==-lookahead):
                order_blocks.append({
                    "type": "bearish",
                    "date": d,
                    "index": i,
                    "zone_low": l,
                    "zone_high": h
                })
    
    return order_blocks


def fibonacci_levels(high_price: float, low_price: float) -> dict:
    """
    Calculate Fibonacci retracement levels between two price points.
    
    Parameters:
        high_price (float): The swing high price.
        low_price (float): The swing low price.
    
    Returns:
        dict: Fibonacci levels with retracement percentages.
    """
    diff = high_price - low_price
    levels = {
        "0.0%": high_price,
        "23.6%": high_price - 0.236 * diff,
        "38.2%": high_price - 0.382 * diff,
        "50.0%": high_price - 0.5 * diff,
        "61.8%": high_price - 0.618 * diff,
        "78.6%": high_price - 0.786 * diff,
        "100.0%": low_price
    }
    return levels

def CoC(df):
    # Ensure date is datetime and sorted for plotting
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

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

    return df

def wick_rejection(candle, threshold=0.6):
    """
    Detect strong lower wick rejection.
    threshold = portion of wick relative to full candle.
    """
    o, h, l, c = candle
    body_low = min(o, c)
    lower_wick = body_low - l
    full_range = h - l
    if full_range == 0:
        return False
    return (lower_wick / full_range) >= threshold

def playbook_long(df_weekly: pd.DataFrame, df_4h: pd.DataFrame):
    """
    Full pipeline for bullish setups.
    """
    signals = []

    # 1. Detect CHoCH+
    df_weekly = CoC(df_weekly)
    choch_indices = list(df_weekly[df_weekly['CoC+']!=0].index)
    OBs = detect_order_blocks(df_weekly)
    OBs = pd.DataFrame(OBs)
    for idx in choch_indices:
        # 2. Find Weekly OB
        if OBs[OBs['index']==idx].empty:
            continue
        ob = OBs[OBs['index']==idx].to_dict()

        # 3. Draw Fib from OB low → swing high (3rd green candle after OB)
        swing_high = df_weekly.loc[idx + 3, "high"]
        fib = fibonacci_levels(high_price=swing_high, low_price=ob['zone_low'])

        fib_618 = fib["50.0%"]

        # 4. Drop to 4H timeframe
        df_zone = df_4h[df_4h["low"] <= fib_618]  # candles touching fib 618
        print(df_zone)
        tapped = False
        for i, row in df_zone.iterrows():
            candle = (row.open, row.high, row.low, row.close)

            if not tapped:
                # first touch = confirmation
                tapped = True
            else:
                # 2nd touch → entry if wick rejection
                if wick_rejection(candle, threshold=0.5):
                    signals.append({
                        "time": row.name,
                        "entry_price": row.close,
                        "stop_loss": ob["low"],
                        "take_profit": swing_high
                    })
    return signals


def main():
    nifty_100 = pd.read_csv("ind_nifty100list.csv")

    for _, row in nifty_100.iterrows():
        ticker = row['Symbol']
        df = pd.read_csv(os.path.join("plots_data", f"{ticker}.csv"))
        df_h = pd.read_csv(os.path.join("hourly_data", f"{ticker}.csv"))
        df_h = df_h.drop(['timestamp', 'gmtoffset'], axis=1)
        df_h['datetime'] = pd.to_datetime(df_h['datetime'])
        df_h.index = df_h['datetime']
        df_h = df_h.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        )
        try:
            signals = playbook_long(df, df_h)
            print("Signals for ",ticker)
            print(signals)
        except:
            continue

main()