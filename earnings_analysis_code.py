import pandas as pd
import os
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Receive file name")
parser.add_argument("--file", required=True)
args = parser.parse_args()

resFile = args.file

base_dir = 'QuarterlyResults'
output_file = f"{resFile.replace('.csv', '')}_analysis.csv"

def fetch_data(code, start, end, interval):
    """Fetch data from EODHD API"""
    try:
        if interval == '1D':
            url = f"https://eodhd.com/api/eod/{code}.NSE?api_token=68be768b173124.42875509&fmt=json&from={start}&to={end}"
        else:
            url = f"https://eodhd.com/api/intraday/{code}.NSE?api_token=68be768b173124.42875509&fmt=json&interval={interval}&from={int(start)}&to={int(end)}"
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        if isinstance(data, dict) and 'error' in data:
            print(f"API Error: {data['error']}")
            return pd.DataFrame()
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching data for {code}: {e}")
        return pd.DataFrame()

def get_trading_window(ts, minutes=15):
    """Get a trading window of specified minutes"""
    t0 = pd.to_datetime(ts)
    t1 = t0 + timedelta(minutes=minutes)
    return t0, t1

def get_20day_avg_volume(secCode, ref_time, slot_duration=15, interval='5m'):
    """Calculate 20-day average volume for the same time slot"""
    avg_volumes = []
    for d in range(1, 21):
        ref_day = ref_time - timedelta(days=d)
        ref_start = ref_day.replace(hour=ref_time.hour, minute=ref_time.minute, second=0)
        ref_end = ref_start + timedelta(minutes=slot_duration)
        
        df = fetch_data(secCode, ref_start.timestamp(), ref_end.timestamp(), interval)
        if not df.empty and 'volume' in df.columns:
            avg_volumes.append(df['volume'].sum())
    
    return pd.Series(avg_volumes).mean() if avg_volumes else 0

results = []
eodhd_df = pd.read_csv('data/EODHD_Codes.csv')

print("Processing", resFile)
results_df = pd.read_csv(resFile)

for _, ann in tqdm(results_df.iterrows(), desc='Processing Announcements', total=len(results_df)):
    try:
        timeFinAnn = ann['Earliest Publish Time']
        annTime = pd.to_datetime(timeFinAnn)
        secCode = eodhd_df[eodhd_df['Security Code'] == ann['sec_code']]['Code'].iloc[0]

        trading_start = annTime.replace(hour=9, minute=15, second=0)
        trading_end = annTime.replace(hour=15, minute=30, second=0)

        # Case A: Announcement during market hours
        if trading_start <= annTime <= trading_end:
            slot_start, slot_end = get_trading_window(annTime, 15)

            # 20-day avg volumes in same slot
            avg_20d_vol = get_20day_avg_volume(secCode, annTime, slot_duration=15)

            # Post-announcement 15min window
            df_post = fetch_data(secCode, slot_start.timestamp(), slot_end.timestamp(), '5m')
            if df_post.empty or len(df_post) < 2:
                continue

            # Volume analysis
            ann_vol = df_post['volume'].sum() if 'volume' in df_post.columns else 0
            ratio = ann_vol / avg_20d_vol if avg_20d_vol > 0 else 0

            # Price movement in 15 min window
            prices_15min = df_post['close'].tolist()
            price_start = df_post['close'].iloc[0]
            price_end = df_post['close'].iloc[-1]
            
            # Bias calculation (price movement direction)
            if price_end >= price_start and ratio > 1.5:
                bias = 'Bullish'
            elif price_end < price_start and ratio > 1.5:
                bias = 'Bearish'

            # Next 2 hours analysis
            conf_start = slot_end
            conf_end = slot_end + timedelta(hours=2)
            df_conf = fetch_data(secCode, conf_start.timestamp(), conf_end.timestamp(), '5m')

            price_change_2h = None
            conf_start_price = None
            conf_end_price = None
            
            if not df_conf.empty:
                conf_start_price = df_conf['close'].iloc[0]
                conf_end_price = df_conf['close'].iloc[-1]
                price_change_2h = (conf_end_price - price_start) / price_start * 100

            results.append({
                "secCode": secCode,
                "annTime": annTime,
                "case": "During Market Hours",
                "ratio": ratio,
                "bias": bias,
                "prices_15min_window": prices_15min,
                "price_start_15min": price_start,
                "price_end_15min": price_end,
                "conf_start_price": conf_start_price,
                "conf_end_price": conf_end_price,
                "next_2h_change_pct": price_change_2h
            })

        # Case B: Announcement outside trading hours
        else:
            next_day = (annTime + timedelta(days=1)).replace(hour=9, minute=30, second=0)
            slot_start = next_day
            slot_end = next_day + timedelta(minutes=15)

            df_next = fetch_data(secCode, slot_start.timestamp(), slot_end.timestamp(), '5m')
            if df_next.empty or len(df_next) < 2:
                continue
            print(df_next)
            # Volume analysis at market open
            ann_vol = df_next['volume'].sum() if 'volume' in df_next.columns else 0
            
            # 20-day avg for same slot (9:30-9:45)
            avg_20d_vol = get_20day_avg_volume(secCode, next_day, slot_duration=15)
            ratio = ann_vol / avg_20d_vol if avg_20d_vol > 0 else 0

            # Gap analysis
            df_daily = fetch_data(secCode, (next_day - timedelta(days=3)).timestamp(), next_day.timestamp(), '1D')

            gap = None
            bias = None
            if len(df_daily) >= 2:
                prev_close = df_daily.iloc[-2]['close']
                today_open = df_daily.iloc[-1]['open']
                print("Prev open and today close", prev_close, today_open)
                gap = (today_open - prev_close) / prev_close * 100
                if gap > 0 and ratio > 1.5:
                    bias = 'bullish'
                elif gap < 0 and ratio > 1.5:
                    bias = 'bearish'
                else:
                    bias = 'None'

            # Price movement in 15 min window at market open
            prices_15min = df_next['close'].tolist()
            price_start = df_next['close'].iloc[0]
            price_end = df_next['close'].iloc[-1]

            # Next 2 hours analysis
            conf_start = slot_end
            conf_end = slot_end + timedelta(hours=2)
            df_conf = fetch_data(secCode, conf_start.timestamp(), conf_end.timestamp(), '5m')

            price_change_2h = None
            conf_start_price = None
            conf_end_price = None
            
            if not df_conf.empty:
                conf_start_price = df_conf['close'].iloc[0]
                conf_end_price = df_conf['close'].iloc[-1]
                price_change_2h = (conf_end_price - price_start) / price_start * 100

            results.append({
                "secCode": secCode,
                "annTime": annTime,
                "case": "Outside Market Hours",
                "gap": gap,
                "ratio": ratio,
                "bias": bias,
                "prices_15min_window": prices_15min,
                "price_start_15min": price_start,
                "price_end_15min": price_end,
                "conf_start_price": conf_start_price,
                "conf_end_price": conf_end_price,
                "next_2h_change_pct": price_change_2h
            })

    except Exception as e:
        print(f"Error processing announcement: {e}")
        continue

# Save results to CSV
if results:
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
    print(f"Processed {len(results)} announcements successfully")
else:
    print("No results to save")
