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
output_file = resFile

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
    # Adjust time for UTC
    t0 = pd.to_datetime(ts) - timedelta(minutes=330)
    t1 = t0 + timedelta(minutes=minutes)
    return t0, t1

print("Processing", resFile)
results_df = pd.read_csv(resFile)

for i, ann in tqdm(results_df.iterrows(), desc='Processing Announcements', total=len(results_df)):
    try:
        timeFinAnn = ann['annTime']
        annTime = pd.to_datetime(timeFinAnn)
        secCode = ann['secCode']

        trading_start = annTime.replace(hour=9, minute=15, second=0)
        trading_end = annTime.replace(hour=15, minute=30, second=0)

        # Case A: Announcement during market hours
        if trading_start <= annTime <= trading_end:
            slot_start, slot_end = get_trading_window(trading_start, 15)

            # Post-announcement 15min window
            df_post = fetch_data(secCode, slot_start.timestamp(), slot_end.timestamp(), '5m')

            # Price movement in 15 min window
            prices_15min = df_post['close'].tolist()

            results_df.loc[i, 'first_15_min_prices'] = str(prices_15min)
        # Case B: Announcement outside trading hours
        else:
            next_day = (annTime + timedelta(days=1)).replace(hour=9, minute=15, second=0)
            slot_start = next_day
            slot_end = next_day + timedelta(minutes=15)

            df_next = fetch_data(secCode, slot_start.timestamp(), slot_end.timestamp(), '5m')
            if df_next.empty or len(df_next) < 2:
                continue

            # Price movement in 15 min window at market open
            prices_15min = df_next['close'].tolist()

            results_df.loc[i, 'first_15_min_prices'] = str(prices_15min)

    except Exception as e:
        print(f"Error processing announcement: {e}")
        continue

# Save results to CSV
results_df.to_csv(output_file, index=False)
print(f"Saved results to {output_file}")
print(f"Processed {len(results_df)} announcements successfully")
