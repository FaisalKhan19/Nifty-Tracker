import pandas as pd
import os
import ta

ohlc_path = "ohlc_weekly_data"

df = pd.read_csv(os.path.join(ohlc_path, "INDUSINDBK_weekly.csv"))
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
df = df.rename(columns={'open': "Open", 'high': 'High', 'low': 'Low', 'close': 'Close'})

# Simple moving average (20-period by default)
df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()

def rma(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1 / period).mean()

def atr(high, low, close, period: int = 14) -> pd.Series:
    # Ref: https://stackoverflow.com/a/74282809/
    prev_close = close.shift()
    tr_all = [high - low, high - prev_close, low - prev_close]
    tr_all = [tr.abs() for tr in tr_all]
    tr = pd.concat(tr_all, axis=1).max(axis=1)
    atr_ = rma(tr, period)
    return atr_

df['ATR_20'] = atr(df['High'], df['Low'], df['Close'], 20)

class SwingStrat:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.df['isSL'] = 0
        self.df['isSH'] = 0
        self.df['detection_time'] = 0
        self.pre_S_type = 'n'
        self.pre_SH = 0
        self.pre_SL = 0
        self.pre_SL_index = -1
        self.pre_SH_index = -1

        ## Variables to store swing lows and swing highs.
        self.SHs = pd.DataFrame(columns=['date_know', 'date_actu', 'SH_value', 'CandleNumber'])
        self.SLs = pd.DataFrame(columns=['date_know', 'date_actu', 'SL_value', 'CandleNumber'])

    ## Function to detect swing lows and swing highs with regards to the instructions described by Lance Beggs.
    def which_S(self, j):
        
        support = (
                    ((self.df['Low'][j-3] < self.df['Low'][j-2]) and (self.df['Low'][j-3] < self.df['Low'][j-1]) and \
                    (self.df['Low'][j-3] < self.df['Low'][j-4]) and (self.df['Low'][j-3] < self.df['Low'][j-5])) or \
                (
                    ((abs(self.df['Low'][j-3]-self.df['Low'][j-4]) < 10) and (self.df['Low'][j-2]>self.df['Low'][j-3])\
                    and (self.df['Low'][j-1]>self.df['Low'][j-3]) and \
                    (self.df['Low'][j-5]>self.df['Low'][j-3]) and (self.df['Low'][j-6]>self.df['Low'][j-3])) or \
                        
                    ((abs(self.df['Low'][j-3]-self.df['Low'][j-4]) < 10) and (abs(self.df['Low'][j-3]-self.df['Low'][j-5]) < 10)\
                    and (self.df['Low'][j-2]>self.df['Low'][j-3]) and (self.df['Low'][j-1]>self.df['Low'][j-3]) and \
                    (self.df['Low'][j-6]>self.df['Low'][j-3]) and (self.df['Low'][j-7]>self.df['Low'][j-3])) or \
                    
                    ((abs(self.df['Low'][j-3]-self.df['Low'][j-5]) < 10) and (self.df['Low'][j-4]>self.df['Low'][j-3]) \
                    and (self.df['Low'][j-2]>self.df['Low'][j-3]) and (self.df['Low'][j-1]>self.df['Low'][j-3]) and \
                    (self.df['Low'][j-6]>self.df['Low'][j-3]) and (self.df['Low'][j-7]>self.df['Low'][j-3]))
                )
        )
        
        resistance = (
                    ((self.df['High'][j-3] > self.df['High'][j-2]) and (self.df['High'][j-3] > self.df['High'][j-1]) and \
                    (self.df['High'][j-3] > self.df['High'][j-4]) and (self.df['High'][j-3] > self.df['High'][j-5])) or \
                (
                    ((abs(self.df['High'][j-3]-self.df['High'][j-4]) < 10) and (self.df['High'][j-2]<self.df['High'][j-3]) and \
                    (self.df['High'][j-1]<self.df['High'][j-3]) and \
                    (self.df['High'][j-5]<self.df['High'][j-3]) and (self.df['High'][j-6]<self.df['High'][j-3])) or \
                        
                    ((abs(self.df['High'][j-3]-self.df['High'][j-4]) < 10) and (abs(self.df['High'][j-3]-self.df['High'][j-5]) < 10) and \
                    (self.df['High'][j-2]<self.df['High'][j-3]) and (self.df['High'][j-1]<self.df['High'][j-3]) and \
                    (self.df['High'][j-6]<self.df['High'][j-3]) and (self.df['High'][j-7]<self.df['High'][j-3])) or \
                    
                    ((abs(self.df['High'][j-3]-self.df['High'][j-5]) < 10) and (self.df['High'][j-4]<self.df['High'][j-3]) and \
                    (self.df['High'][j-2]<self.df['High'][j-3]) and (self.df['High'][j-1]<self.df['High'][j-3]) and \
                    (self.df['High'][j-6]<self.df['High'][j-3]) and (self.df['High'][j-7]<self.df['High'][j-3]))
                )
        )
        if support:
            if self.pre_S_type=='n' or self.pre_S_type=='H':
                pre_S_type = 'L'
                pre_SL = self.df['Low'][j-3]
                pre_SL_index = j-3
                self.df['isSL'][j-3] = 1
                to_append = [self.df['date'][j], self.df['date'][j-3], self.df['Low'][j-3], j-3]
                SLs_length = len(self.SLs)
                self.SLs.loc[SLs_length] = to_append
                self.df['detection_time'][j-3] = self.df['date'][j]
                        
            elif(self.pre_S_type=='L' and self.df['Low'][j-3]<pre_SL):
                pre_SL = self.df['Low'][j-3]
                self.df['isSL'][pre_SL_index] = 0
                self.df['detection_time'][pre_SL_index] = 0
                self.df['isSL'][j-3] = 1
                self.df['detection_time'][j-3] = self.df['date'][j]
                self.SLs['SL_value'][self.df.index[pre_SL_index]] = self.df['Low'][j-3]
                self.SLs['CandleNumber'][self.df.index[pre_SL_index]] = j-3
                pre_SL_index = j-3       
        #####################################################################################################        
        elif resistance:
            if self.pre_S_type=='n' or self.pre_S_type=='L':
                self.pre_S_type = 'H'
                self.pre_SH = self.df['High'][j-3]
                self.pre_SH_index = j-3
                self.df['isSH'][j-3] = 1
                to_append = [self.df['date'][j], self.df['date'][j-3], self.df['High'][j-3], j-3]
                SHs_length = len(self.SHs)
                self.SHs.loc[SHs_length] = to_append
                self.df['detection_time'][j-3] = self.df['date'][j]
                
            elif (self.pre_S_type=='H' and self.df['High'][j-2]>self.pre_SH):
                self.pre_SH = self.df['High'][j-3]
                self.df['isSH'][self.pre_SH_index] = 0
                self.df['detection_time'][self.pre_SH_index] = 0
                self.df['isSH'][j-3] = 1
                self.df['detection_time'][j-3] = self.df['date'][j]
                self.SHs['SH_value'][self.df.index[self.pre_SH_index]] = self.df['High'][j-3]
                self.SHs['CandleNumber'][self.df.index[self.pre_SH_index]] = j-3
                self.pre_SH_index = j-3
                
    def addSwing(self, lookback=10, lookahead=3, proximity_pct=0.07):
        ## This loop adds swing highs and swing lows with their detection time to the original dataframe.
        for i in range(8, self.df.shape[0]-2):
            self.which_S(i)

        # Detect swing highs/lows
        swing_highs = []
        swing_lows = []
        last_sh_index = None
        last_sl_index = None

        for i in range(lookback, len(self.df) - lookahead):
            high = self.df.loc[i, "High"]
            low = self.df.loc[i, "Low"]

            if high == self.df["High"].iloc[i - lookback : i + lookahead + 1].max():
                record_high = False
                if last_sh_index is None or (i - last_sh_index) >= 10:
                    record_high = True
                else:
                    if not swing_highs or abs(high - swing_highs[-1][1]) / swing_highs[-1][1] > proximity_pct:
                        record_high = True
                if record_high:
                    swing_highs.append((self.df.loc[i, "date"], high))
                    last_sh_index = i
            elif low == self.df["Low"].iloc[i - lookback : i + lookahead + 1].min():
                record_low = False
                if last_sl_index is None or (i - last_sl_index) >= 10:
                    record_low = True
                else:
                    if not swing_lows or abs(low - swing_lows[-1][1]) / swing_lows[-1][1] > proximity_pct:
                        record_low = True
                if record_low:
                    swing_lows.append((self.df.loc[i, "date"], low))
                    last_sl_index = i

        # Convert to DataFrame
        SHs = pd.DataFrame(swing_highs, columns=["date", "SH_value"])
        SLs = pd.DataFrame(swing_lows, columns=["date", "SL_value"])

        # Make sure date types align
        self.df['date'] = pd.to_datetime(self.df['date'])
        SHs['date'] = pd.to_datetime(SHs['date'])
        SLs['date'] = pd.to_datetime(SLs['date'])

        # Merge with left join
        self.df = self.df.merge(SHs, on='date', how='left')
        self.df = self.df.merge(SLs, on='date', how='left')

        # Optional: add flags
        self.df['isSH'] = self.df['SH_value'].notna().astype(int)
        self.df['isSL'] = self.df['SL_value'].notna().astype(int)

    def addFVG(self, alpha=0.2):
        return


    def FindTrend(self, i):
       return 

strat = SwingStrat(df)
strat.addSwing()
print(strat.df.columns)