import numpy as np
import pandas as pd
import os

def detect_order_blocks(df: pd.DataFrame, lookahead: int = 3, displacement_factor: float = 0.075):
    """
    Robust detection of bullish/bearish OBs (last red before lookahead greens, etc).
    Returns list of dicts with fields: type, date, index, zone_low, zone_high
    """
    df = df.copy().reset_index(drop=True)
    required = ['open','high','low','close']
    if not all(col in df.columns for col in required):
        raise ValueError(f"detect_order_blocks needs columns: {required}")

    # add date if missing
    if 'date' not in df.columns:
        df['date'] = df.index

    df['candle_type'] = np.where(df['open'] > df['close'], -1, 1)
    avg_body = (df['close'] - df['open']).abs().rolling(20, min_periods=1).mean()

    obs = []
    n = len(df)
    for i in range(n - lookahead):
        d = df.at[i, 'date']
        o = df.at[i, 'open']
        h = df.at[i, 'high']
        l = df.at[i, 'low']
        c = df.at[i, 'close']

        forward = df.iloc[i+1 : i+1+lookahead]
        if len(forward) < lookahead:
            continue

        # bullish OB: current candle red, next lookahead candles all green and closing above current close
        if c < o:
            if (forward['candle_type'] == 1).all() and (forward['close'] > c).all():
                drive_close = forward['close'].iloc[-1]
                # either ratio displacement or large body relative to rolling avg
                cond1 = (drive_close / c - 1) > displacement_factor
                cond2 = (abs(drive_close - forward['open'].iloc[-1]) > displacement_factor * max(1e-9, avg_body.iloc[i]))
                if cond1 or cond2:
                    obs.append({
                        "type": "bullish",
                        "date": d,
                        "index": i,
                        "zone_low": float(l),
                        "zone_high": float(h),
                    })

        # bearish OB: current candle green, next lookahead candles all red and closing below current close
        if c > o:
            if (forward['candle_type'] == -1).all() and (forward['close'] < c).all():
                drive_close = forward['close'].iloc[-1]
                cond1 = (c / max(1e-9, drive_close) - 1) > displacement_factor
                cond2 = (abs(drive_close - forward['open'].iloc[-1]) > displacement_factor * max(1e-9, avg_body.iloc[i]))
                if cond1 or cond2:
                    obs.append({
                        "type": "bearish",
                        "date": d,
                        "index": i,
                        "zone_low": float(l),
                        "zone_high": float(h),
                    })
    return obs

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

def fibonacci_levels(high_price: float, low_price: float) -> dict:
    diff = high_price - low_price
    return {
        "0.0%": float(high_price),
        "23.6%": float(high_price - 0.236 * diff),
        "38.2%": float(high_price - 0.382 * diff),
        "50.0%": float(high_price - 0.5 * diff),
        "61.8%": float(high_price - 0.618 * diff),
        "78.6%": float(high_price - 0.786 * diff),
        "100.0%": float(low_price),
    }

def wick_rejection(candle, threshold=0.6):
    o, h, l, c = candle
    body_low = min(o, c)
    lower_wick = body_low - l
    full_range = h - l
    if full_range == 0:
        return False
    return (lower_wick / full_range) >= threshold

def check_outcome(df_future: pd.DataFrame, stop_loss: float, take_profit: float, horizon: int = 8, freq: str = "W"):
    """
    Walk forward into df_future (weekly candles).
    Return 'SL' if stop loss is hit first, 'TP' if take profit is hit first,
    or 'NONE' if neither is hit within horizon.
    """
    # only look horizon weeks ahead
    df_future = df_future.iloc[:horizon]
    for _, row in df_future.iterrows():
        low, high = row['low'], row['high']
        if low <= stop_loss and high >= take_profit:
            # Both touched same candle → assume worst case (SL first) unless you want optimistic
            return "SLTP"
        elif low <= stop_loss:
            return "SL"
        elif high >= take_profit:
            return "TP"
    return "NONE"


def playbook_long(df_weekly: pd.DataFrame, df_4h: pd.DataFrame):
    """
    Full pipeline for bullish setups aligned to the PDF playbook:
    - Weekly OB + CoC must exist
    - Fib anchored OB -> 3rd green (use 61.8% for 0.6 retrace)
    - Find 4H OB inside that fib zone and inside the weekly OB
    - Wait for 1st tap, then 2nd tap -> if wick rejection on 2nd tap, enter
    """
    signals = []

    # Coerce inputs
    df_weekly = df_weekly.copy().reset_index(drop=True)
    df_4h = df_4h.copy()
    # Ensure 4H index is datetime and sorted
    if not isinstance(df_4h.index, pd.DatetimeIndex):
        if 'datetime' in df_4h.columns:
            df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])
            df_4h = df_4h.set_index('datetime').sort_index()
        elif 'date' in df_4h.columns:
            df_4h['datetime'] = pd.to_datetime(df_4h['date'])
            df_4h = df_4h.set_index('datetime').sort_index()
        else:
            df_4h.index = pd.to_datetime(df_4h.index)
            df_4h = df_4h.sort_index()

    # 1. Mark CoC on weekly (CoC function must have been run or be available)
    df_weekly = CoC(df_weekly)   # Ensure your swings are precomputed (shl_* columns)
    choch_indices = list(df_weekly[df_weekly['CoC+'] != 0].index)

    # detect OBs
    weekly_obs = pd.DataFrame(detect_order_blocks(df_weekly, lookahead=3))
    fourh_obs = pd.DataFrame(detect_order_blocks(df_4h.reset_index(drop=True), lookahead=3))

    for idx in choch_indices:
        # 2. Find Weekly OB that matches the CHoCH index
        ob_rows = weekly_obs[(weekly_obs['index'] >= int(idx)) & (weekly_obs['index'] < (int(idx)+4))]
        if ob_rows.empty:
            continue

        # pick first matching weekly OB
        weekly_ob = ob_rows.iloc[0].to_dict()
        initiation_date = ob_rows['date'].iloc[0]
        # ensure swing high exists (3rd green after OB) — guard against OOB
        if idx + 3 >= len(df_weekly):
            continue
        swing_high = float(df_weekly.loc[idx + 3, "high"])
        fib = fibonacci_levels(high_price=swing_high, low_price=float(df_weekly.loc[idx, 'low']))
        fib_618 = fib["50.0%"]
        fib_100 = fib["100.0%"]

        # 3. Identify 4H OBs fully inside the weekly fib 0.6-1.0 zone and inside weekly OB boundaries
        if not fourh_obs.empty:
            candidates = fourh_obs[
                (fourh_obs['zone_low'] >= fib_100) & (fourh_obs['zone_high'] <= fib_618)
            ]
            #keep only 4H OBs that form after initiation date
            candidates = candidates[candidates['date'] >= initiation_date]
        else:
            candidates = pd.DataFrame([])

        # If no 4H OBs, as fallback consider any 4H candles touching the fib zone
        if candidates.empty:
            df_touch = df_4h[(df_4h['low'] >= fib_100) & (df_4h['high'] <= fib_618)].sort_index()
            if not df_touch.empty:
                df_touch = df_touch[df_touch.index >= initiation_date]
            tapped = False
            for t_idx, row in df_touch.iterrows():
                if not tapped:
                    tapped = True
                else:
                    if wick_rejection((row['open'], row['high'], row['low'], row['close']), threshold=0.5):
                        outcome_6 = check_outcome(df_weekly.loc[idx+1:], weekly_ob['zone_low'], swing_high, horizon=6)
                        outcome_8 = check_outcome(df_weekly.loc[idx+1:], weekly_ob['zone_low'], swing_high, horizon=8)

                        signals.append({
                            "time": t_idx,
                            "entry_price": float(row['close']),
                            "stop_loss": float(weekly_ob['zone_low']),
                            "take_profit": swing_high,
                            "outcome_6": outcome_6,
                            "outcome_8": outcome_8,
                            "initiation_date": initiation_date,
                        })
                        break
            continue

        # For each 4H OB candidate, check taps inside that OB zone
        for _, fob in candidates.iterrows():
            ob_low = float(fob['zone_low'])
            ob_high = float(fob['zone_high'])
            # 4H candles overlapping this 4H OB (in time order)
            df_touch = df_4h[(df_4h['low'] <= ob_high) & (df_4h['high'] >= ob_low)].sort_index()
            tap_count = 0
            for t_idx, row in df_touch.iterrows():
                if tap_count == 0:
                    tap_count = 1
                    continue
                else:
                    # 2nd tap — require lower-wick rejection (bullish)
                    if wick_rejection((row['open'], row['high'], row['low'], row['close']), threshold=0.5):
                        outcome_6 = check_outcome(df_weekly.loc[idx+1:], ob_low, swing_high, horizon=6)
                        outcome_8 = check_outcome(df_weekly.loc[idx+1:], weekly_ob['zone_low'], swing_high, horizon=8)

                        signals.append({
                            "time": t_idx,
                            "entry_price": float(row['close']),
                            "stop_loss": ob_low,     # 4H OB low
                            "take_profit": swing_high,
                            "outcome_6": outcome_6,
                            "outcome_8": outcome_8,
                            "initiation_date": initiation_date,
                        })
                        break

    return signals

def main():
    nifty_100 = pd.read_csv("ind_nifty100list.csv")
    signals = []
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
        this_signals = playbook_long(df, df_h)
        if this_signals:
            for signal in this_signals:
                signal.update({"ticker": ticker})
                signals.append(signal)
    if signals:
        pd.DataFrame(signals).to_csv("all_signals.csv", index=False)
main()