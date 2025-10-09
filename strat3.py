import numpy as np
import pandas as pd
import os
from smartmoneyconcepts.smc import smc
from strat_utils import _merge_with_prefix
from tqdm import tqdm

def detect_fvg(df: pd.DataFrame):
    """
    Detect Fair Value Gaps (FVG) in the dataframe.
    Returns a boolean series indicating FVG presence.
    """
    df = df.copy()
    fvg_bullish = np.zeros(len(df), dtype=bool)
    fvg_bearish = np.zeros(len(df), dtype=bool)
    
    for i in range(2, len(df)):
        # Bullish FVG: gap between candle[i-2].high and candle[i].low
        if df.iloc[i-2]['high'] < df.iloc[i]['low']:
            fvg_bullish[i-1] = True
            
        # Bearish FVG: gap between candle[i-2].low and candle[i].high  
        if df.iloc[i-2]['low'] > df.iloc[i]['high']:
            fvg_bearish[i-1] = True
    
    return fvg_bullish | fvg_bearish

def detect_order_blocks(df: pd.DataFrame, lookahead: int = 3, displacement_factor: float = 0.075):
    """
    Robust detection of bullish/bearish OBs (last red before lookahead greens, etc).
    Now requires FVG formation in the lookahead period.
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
    
    # Detect FVGs
    fvg_present = detect_fvg(df)

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
        
        # Check if FVG is formed in the lookahead period
        fvg_in_lookahead = fvg_present[i+1:i+1+lookahead].any()
        if not fvg_in_lookahead:
            continue

        # bullish OB: current candle red, next lookahead candles all green and closing above current close
        if c < o:
            if ((forward['candle_type'] == 1).sum() >= 2) and (forward['close'] > c).all():
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
            if (forward['candle_type'] == -1).sum() >= 2 and (forward['close'] < c).all():
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

def CoC(df: pd.DataFrame):
    # Ensure date is datetime and sorted for plotting
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

    bullish_pattern = [-1, 1, -1, 1]  # Bullish CoC+
    bearish_pattern = [1, -1, 1, -1]  # Bearish CoC+

    if 'shl_HighLow' not in df.columns:
        shl = smc.swing_highs_lows(df, swing_length=3)
        df = _merge_with_prefix(df, shl, 'shl')

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

def wick_rejection_bullish(candle, threshold=0.25):
    """Check for bullish wick rejection (strong lower wick)"""
    o, h, l, c = candle
    body_low = min(o, c)
    lower_wick = body_low - l
    full_range = h - l
    if full_range == 0:
        return False
    return (lower_wick / full_range) >= threshold

def wick_rejection_bearish(candle, threshold=0.25):
    """Check for bearish wick rejection (strong upper wick)"""
    o, h, l, c = candle
    body_high = max(o, c)
    upper_wick = h - body_high
    full_range = h - l
    if full_range == 0:
        return False
    return (upper_wick / full_range) >= threshold

def detect_swing_structure_break(df: pd.DataFrame, start_idx: int, direction: str):
    """
    Detect break in swing structure for exit conditions.
    For LONG: Look for lower swing low AND lower swing high
    For SHORT: Look for higher swing low AND higher swing high
    Returns (broken, exit_price) tuple
    """
    if 'shl_HighLow' not in df.columns or 'shl_Level' not in df.columns:
        return False, None, None
    
    # Get swings after entry point
    swing_data = df.iloc[start_idx:].copy()
    swing_highs = swing_data[swing_data['shl_HighLow'] == 1]
    swing_lows = swing_data[swing_data['shl_HighLow'] == -1]
    
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return False, None, None
    
    if direction == "LONG":
        # Check for lower swing low AND lower swing high
        recent_lows = swing_lows['shl_Level'].tail(2)
        recent_highs = swing_highs['shl_Level'].tail(2)
        if (recent_lows.iloc[-1] < 1.05*recent_lows.iloc[-2] and 
            recent_highs.iloc[-1] < 1.05*recent_highs.iloc[-2]):
            # Exit at the most recent close price
            date, close = df.iloc[max(swing_lows.index[-1], swing_highs.index[-1])][['date', 'close']]
            return True, date, close
            
    elif direction == "SHORT":
        # Check for higher swing low AND higher swing high
        recent_lows = swing_lows['shl_Level'].tail(2)
        recent_highs = swing_highs['shl_Level'].tail(2)
        
        if (recent_lows.iloc[-1] > 0.95*recent_lows.iloc[-2] and 
            recent_highs.iloc[-1] > 0.95*recent_highs.iloc[-2]):
            # Exit at the most recent close price
            date, close = df.iloc[max(swing_lows.index[-1], swing_highs.index[-1])][['date', 'close']]
            return True, date, close
    
    return False, None, None

def check_outcome_long(df_future: pd.DataFrame, stop_loss: float, take_profit: float, horizon: int = 8):
    """
    Walk forward into df_future for LONG trades.
    Return 'SL' if stop loss is hit, 'Structure Break' if swing structure breaks,
    or 'NONE' if neither condition is met within horizon.
    """
    df_future = df_future.iloc[:horizon].copy()
    
    # Ensure swing high/low data exists
    if 'shl_HighLow' not in df_future.columns:
        shl = smc.swing_highs_lows(df_future.reset_index(), swing_length=3)
        df_future = _merge_with_prefix(df_future.reset_index(), shl, 'shl').set_index('index')
    structure_broken, date, exit_price = detect_swing_structure_break(df_future, 0, "LONG")
    
    for i, (idx, row) in enumerate(df_future.iterrows()):
        low = row['low']
        high = row['high']
        date = row['date']
        # Check stop loss
        if structure_broken and date<=row['date']:
            return "Structure Break", exit_price, date
        if low <= stop_loss:
            return "SL", None, date
        if high >=take_profit:
            return "TP", None, date
    return "Max_Hold", df_future.iloc[-1]["close"], df_future.iloc[-1]["date"]

def check_outcome_short(df_future: pd.DataFrame, stop_loss: float, take_profit: float, horizon: int = 8):
    """
    Walk forward into df_future for SHORT trades.
    Return 'SL' if stop loss is hit, 'Structure Break' if swing structure breaks,
    or 'NONE' if neither condition is met within horizon.
    """
    df_future = df_future.iloc[:horizon].copy()
    
    # Ensure swing high/low data exists
    if 'shl_HighLow' not in df_future.columns:
        shl = smc.swing_highs_lows(df_future.reset_index(), swing_length=3)
        df_future = _merge_with_prefix(df_future.reset_index(), shl, 'shl').set_index('index')

    structure_broken, date, exit_price = detect_swing_structure_break(df_future, 0, "SHORT")
    for i, (idx, row) in enumerate(df_future.iterrows()):
        high = row['high']
        low = row['low']
        date = row['date']
        if structure_broken and date>row['date']:
            return "Structure Break", exit_price, date
        # Check stop loss
        if high >= stop_loss:
            return "SL", None, date
        if low <= take_profit:
            return "TP", None, date
    return "Max_Hold", df_future.iloc[-1]["close"], df_future.iloc[-1]["date"]

def playbook_long(df_weekly: pd.DataFrame, df_4h: pd.DataFrame):
    """
    Full pipeline for bullish setups aligned to the playbook:
    - Weekly OB + bullish CoC must exist
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

    # 1. Mark CoC on weekly
    df_weekly = CoC(df_weekly)
    df_4h = CoC(df_4h)

    bullish_choch_indices = list(df_weekly[df_weekly['CoC+'] == 1].index)

    # Detect OBs
    weekly_obs = pd.DataFrame(detect_order_blocks(df_weekly, lookahead=3))
    fourh_obs = pd.DataFrame(detect_order_blocks(df_4h.reset_index(drop=True), lookahead=3))
    fourh_obs['date'] = pd.to_datetime(fourh_obs['date'])
    # Filter for bullish weekly OBs only
    if not weekly_obs.empty:
        weekly_obs = weekly_obs[weekly_obs['type'] == 'bullish']
    
    # Filter for bullish 4H OBs only
    if not fourh_obs.empty:
        fourh_obs = fourh_obs[fourh_obs['type'] == 'bullish']

    for idx in bullish_choch_indices:
        # 2. Find Weekly bullish OB that matches the CHoCH index
        ob_rows = weekly_obs[(weekly_obs['index'] >= int(idx)) & (weekly_obs['index'] < (int(idx)+4))]
        if ob_rows.empty:
            continue

        # Pick first matching weekly OB
        weekly_ob = ob_rows.iloc[0].to_dict()
        initiation_date = ob_rows['date'].iloc[0]
        
        # Ensure swing high exists (3rd green after OB)
        if idx + 3 >= len(df_weekly):
            continue

        match = df_weekly.loc[df_weekly['date'] == initiation_date + pd.Timedelta(weeks=3), "high"]
        swing_high = float(match.iloc[0]) if not match.empty else None
        if not swing_high:
            continue
        fib = fibonacci_levels(high_price=swing_high, low_price=float(weekly_ob['zone_low']))
        fib_100 = fib["100.0%"]
        fib_50 = fib["50.0%"]
        
        slice = df_4h[df_4h.index>initiation_date+pd.Timedelta(weeks=3)]
        df_touch = slice[(slice['low'] >= fib_100) & (slice['close'] <= fib_50)].sort_index()

        for t_idx, row in df_touch.iterrows():
            if wick_rejection_bullish((row['open'], row['high'], row['low'], row['close'])) or True:
                sl = min(row['close']*0.975, weekly_ob['zone_low']*0.975)
                tp = swing_high*1.025
                # Get index position for df_4h
                entry_idx = df_4h.index.get_loc(t_idx)
                future_data = df_4h.iloc[entry_idx+1:]
                outcome_8, exit_close, exit_date = check_outcome_long(future_data.reset_index(drop=True), sl, tp, horizon=8*42)
                
                if exit_close:
                    returns = (exit_close/row['close'] - 1)
                else:
                    if outcome_8 == 'SL':
                        returns = (sl/row['close'] - 1)
                    if outcome_8 == 'TP':
                        returns = (tp/row['close'] - 1)
                    elif exit_close:
                        returns = (exit_close/row['close'] - 1)
                if returns<0:
                    returns = max((sl/row['close'] - 1), returns)
                signals.append({
                    "setup_type": "LONG",
                    "time": t_idx,
                    "entry_price": float(row['close']),
                    "stop_loss": sl,
                    "take_profit": tp,  # Keep for reference but not used in exit logic
                    "outcome_8": outcome_8,
                    "initiation_date": initiation_date,
                    "returns": returns,
                    "exit_time": exit_date
                })
                break
    return signals

def playbook_short(df_weekly: pd.DataFrame, df_4h: pd.DataFrame):
    """
    Full pipeline for bearish setups (mirror of long logic):
    - Weekly bearish OB + bearish CoC must exist
    - Fib anchored OB high -> swing low (3rd red candle)
    - Find 4H bearish OB inside that fib zone
    - Wait for 1st tap, then 2nd tap -> if upper wick rejection, enter short
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

    # 1. Mark CoC on weekly
    df_weekly = CoC(df_weekly)
    df_4h = CoC(df_4h)
    bearish_choch_indices = list(df_weekly[df_weekly['CoC+'] == -1].index)

    # Detect OBs
    weekly_obs = pd.DataFrame(detect_order_blocks(df_weekly, lookahead=3))
    fourh_obs = pd.DataFrame(detect_order_blocks(df_4h.reset_index(drop=True), lookahead=3))
    fourh_obs['date'] = pd.to_datetime(fourh_obs['date'])
    # Filter for bearish weekly OBs only
    if not weekly_obs.empty:
        weekly_obs = weekly_obs[weekly_obs['type'] == 'bearish']
    
    # Filter for bearish 4H OBs only
    if not fourh_obs.empty:
        fourh_obs = fourh_obs[fourh_obs['type'] == 'bearish']

    for idx in bearish_choch_indices:
        # 2. Find Weekly bearish OB that matches the CHoCH index
        ob_rows = weekly_obs[(weekly_obs['index'] >= int(idx)) & (weekly_obs['index'] < (int(idx)+4))]
        if ob_rows.empty:
            continue

        # Pick first matching weekly OB
        weekly_ob = ob_rows.iloc[0].to_dict()
        initiation_date = ob_rows['date'].iloc[0]
        
        # Ensure swing low exists (3rd red after OB)
        if idx + 3 >= len(df_weekly):
            continue

        match = df_weekly.loc[df_weekly['date'] == initiation_date + pd.Timedelta(weeks=3), "low"]
        swing_low = float(match.iloc[0]) if not match.empty else None
        if not swing_low:
            continue
        fib = fibonacci_levels(high_price=swing_low, low_price=float(weekly_ob['zone_high']))
        fib_100 = fib["100.0%"]  # This is now the lower bound
        fib_50 = fib["50.0%"]

        slice = df_4h[df_4h.index>initiation_date+pd.Timedelta(weeks=3)]
        df_touch = slice[(slice['close'] >= fib_50) & (slice['high'] <= fib_100)].sort_index()

        for t_idx, row in df_touch.iterrows():
            if wick_rejection_bearish((row['open'], row['high'], row['low'], row['close'])) or True:
                sl = max(row['close']*1.025, weekly_ob['zone_high']*1.025)
                tp = swing_low*0.975
                # Get index position for df_4h
                entry_idx = df_4h.index.get_loc(t_idx)
                future_data = df_4h.iloc[entry_idx+1:]
                outcome_8, exit_close, exit_date = check_outcome_short(future_data.reset_index(drop=True), sl, tp, horizon=8*42)
                
                if exit_close:
                    returns = (row['close']/exit_close - 1)
                else:
                    if outcome_8 == 'SL':
                        returns = (row['close']/sl - 1)
                    if outcome_8 == 'TP':
                        returns = (row['close']/tp - 1)
                    elif exit_close:
                        returns = (row['close']/exit_close - 1)
                if returns<0:
                    returns = max((row['close']/sl - 1), returns)
                signals.append({
                    "setup_type": "SHORT",
                    "time": t_idx,
                    "entry_price": float(row['close']),
                    "stop_loss": sl,
                    "take_profit": tp,  # Keep for reference but not used in exit logic
                    "outcome_8": outcome_8,
                    "initiation_date": initiation_date,
                    "returns": returns,
                    "exit_time": exit_date
                })
                break
    return signals

def main():
    """
    Main execution function that processes all tickers and generates both long and short signals
    """
    nifty_100 = pd.read_csv("ind_nifty100list.csv")
    all_signals = []
    
    for _, row in tqdm(nifty_100.iterrows(), desc="Processing:"):
        ticker = row['Symbol']
        # if ticker != 'INDHOTEL':
        #     continue
        # Load data
        df_weekly = pd.read_csv(os.path.join("plots_data", f"{ticker}.csv"))
        df_hourly = pd.read_csv(os.path.join("hourly_data", f"{ticker}.csv"))
        
        # Process hourly to 4H
        df_hourly = df_hourly.drop(['timestamp', 'gmtoffset'], axis=1, errors='ignore')
        df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
        df_hourly.index = df_hourly['datetime']
        df_4h = df_hourly.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        
        # Get long signals
        long_signals = playbook_long(df_weekly.copy(), df_4h.copy())
        for signal in long_signals:
            signal.update({"ticker": ticker})
            all_signals.append(signal)
        
        # Get short signals  
        short_signals = playbook_short(df_weekly.copy(), df_4h.copy())
        for signal in short_signals:
            signal.update({"ticker": ticker})
            all_signals.append(signal)
                
    
    # Save all signals
    if all_signals:
        signals_df = pd.DataFrame(all_signals)
        signals_df.to_csv("all_signals_long_short.csv", index=False)
        print(f"\nGenerated {len(all_signals)} total signals:")
        print(f"Long signals: {len([s for s in all_signals if s['setup_type'] == 'LONG'])}")
        print(f"Short signals: {len([s for s in all_signals if s['setup_type'] == 'SHORT'])}")
        
        # Performance summary
        # outcomes_6 = signals_df['outcome_6'].value_counts()
        outcomes_8 = signals_df['outcome_8'].value_counts()
        # print(f"\n6-week outcomes: {dict(outcomes_6)}")
        print(f"8-week outcomes: {dict(outcomes_8)}")
    else:
        print("No signals generated.")

if __name__ == "__main__":
    main()