import pandas as pd
import os
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Recieve file name")
parser.add_argument("--file", required=True)
args = parser.parse_args()

resFile = args.file

base_dir = 'QuarterlyResults'
output_file = f"{resFile}_analysis.csv"

def fetch_data(code, start, end, interval):
    if interval == '1D':
        url = f"https://eodhd.com/api/eod/{code}.NSE?api_token=68be768b173124.42875509&fmt=json&from={start}&to={end}"
    else:
        url = f"https://eodhd.com/api/intraday/{code}.NSE?api_token=68be768b173124.42875509&fmt=json&interval={interval}&from={int(start)}&to={int(end)}"
    res = requests.get(url)
    data = res.json()
    return pd.DataFrame(data)

def get_trading_window(ts, minutes=15):
    t0 = pd.to_datetime(ts)
    t1 = t0 + timedelta(minutes=minutes)
    return t0, t1

results = []
eodhd_df = pd.read_csv('data/EODHD_Codes.csv')

print("Processing", resFile)
results_df = pd.read_csv(resFile)

for _, ann in tqdm(results_df.iterrows(), desc='Processing Announcements', total=len(results_df)):
    # try:
    timeFinAnn = ann['Earliest Publish Time']
    annTime = pd.to_datetime(timeFinAnn)
    secCode = eodhd_df[eodhd_df['Security Code'] == ann['sec_code']]['Code'].iloc[0]

    trading_start = annTime.replace(hour=9, minute=15, second=0)
    trading_end   = annTime.replace(hour=15, minute=30, second=0)

    # Case A: Announcement during market hours
    if trading_start <= annTime <= trading_end:
        slot_start, slot_end = get_trading_window(annTime, 15)

        # 20-day avg volumes in same slot
        avg_volumes = []
        for d in range(1, 21):
            ref_day = annTime - timedelta(days=d)
            ref_start = ref_day.replace(hour=slot_start.hour, minute=slot_start.minute)
            ref_end   = ref_day.replace(hour=slot_end.hour, minute=slot_end.minute)
            df = fetch_data(secCode, ref_start.timestamp(), ref_end.timestamp(), '5m')
            if not df.empty:
                avg_volumes.append(df['volume'].sum())

        avg_20d_vol = pd.Series(avg_volumes).mean() if avg_volumes else 0

        # Post-announcement 15min window
        df_post = fetch_data(secCode, slot_start.timestamp(), slot_end.timestamp(), '5m')
        if df_post.empty:
            continue
        print("price data 5 min after", df_post)
        ann_vol = df_post['volume'].sum()
        ratio = ann_vol / avg_20d_vol if avg_20d_vol else 0

        bias_change_1 = df_post['close'].iloc[-1] - df_post['close'].iloc[0]
        bias_change_2 = df_post['close'].iloc[-2] - df_post['close'].iloc[-1]
        bias_change_3 = df_post['close'].iloc[-3] - df_post['close'].iloc[-2]
        bias = "Bullish" if (bias_change_1 and bias_change_2 and bias_change_3) > 0 else "Bearish"

        # Next 2 hours
        conf_start = slot_end
        conf_end   = slot_end + timedelta(hours=2)
        df_conf = fetch_data(secCode, conf_start.timestamp(), conf_end.timestamp(), '5m')

        price_change = None
        if not df_conf.empty:
            price_change = (df_conf['close'].iloc[-1] - df_post['close'].iloc[0]) / df_post['close'].iloc[0] * 100

        results.append({
            "secCode": secCode,
            "annTime": annTime,
            "ratio": ratio,
            "bias": bias,
            "next_2h_change_pct": price_change
        })

    # Case B: Announcement outside trading hours
    else:
        next_day = (annTime + timedelta(days=1)).replace(hour=9, minute=30, second=0)
        slot_start = next_day
        slot_end   = next_day + timedelta(minutes=15)

        df_next = fetch_data(secCode, slot_start.timestamp(), slot_end.timestamp(), '5m')
        if df_next.empty:
            continue

        ann_vol = df_next['volume'].sum()

        # 20-day avg slot vols
        avg_volumes = []
        for d in range(1, 21):
            ref_day = next_day - timedelta(days=d)
            ref_start = ref_day.replace(hour=9, minute=30, second=0)
            ref_end   = ref_start + timedelta(minutes=15)
            df = fetch_data(secCode, ref_start.timestamp(), ref_end.timestamp(), '5m')
            if not df.empty:
                avg_volumes.append(df['volume'].sum())

        avg_20d_vol = pd.Series(avg_volumes).mean() if avg_volumes else 0
        ratio = ann_vol / avg_20d_vol if avg_20d_vol else 0

        # Daily candles for gap check
        df_daily = fetch_data(secCode, (next_day - timedelta(days=3)).timestamp(), next_day.timestamp(), '1D')

        if len(df_daily) >= 2:
            prev_close = df_daily.iloc[-2]['close']
            today_open = df_daily.iloc[-1]['open']
            gap = today_open - prev_close
            bias = "Bullish" if gap > 0 else "Bearish"
        else:
            bias = None

        # Optional: track next 2h change on that day
        conf_start = slot_end
        conf_end   = slot_end + timedelta(hours=2)
        df_conf = fetch_data(secCode, conf_start.timestamp(), conf_end.timestamp(), '5m')

        price_change = None
        if not df_conf.empty:
            price_change = (df_conf['close'].iloc[-1] - df_next['close'].iloc[0]) / df_next['close'].iloc[0] * 100

        results.append({
            "secCode": secCode,
            "annTime": annTime,
            "ratio": ratio,
            "bias": bias,
            "next_2h_change_pct": price_change
        })

    # except Exception as e:
    #     print("Error processing", ann, e)

# Save results to CSV
df_out = pd.DataFrame(results)
df_out.to_csv(output_file, index=False)
print(f"Saved results to {output_file}")
